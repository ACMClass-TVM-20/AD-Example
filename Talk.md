## 1

Hi guys, now let me introduce the AD Pass.

This document contains five parts now and I will take about 20 mins.

The first part is an overview of this Pass. Let us first look this example. The Before module has a single function `main`. The body is just a simple add, and sum the Tensor to a scalar `gv0`. After applying AD, the After module has two functions: `main` is the same with Before, and `main_adjoint` is the new adjoint function.

The `main_adjoint` is copied from `main`, and AD will do some modifications. To be detailed, we will add some extra bindings after the original body, and the return value is different.

These bindings are exactly the differentiation calculation. And we can find that they are in a reverse order of the original bindings. You see, first `gv0_adjoint` and then `lv0_adjoint` and finally the adjoint of inputs.

`main_adjoint` returns a pair, where the first is the original return value of `main`, and the second is an array contains adjoints we need. We have an argument in our Pass, named `require_grads`. By default it is None, and we will calculate adjoints for all inputs of function, in this case, x and y. If `require_grads` is a list containing single `x`, we will only calculate and return the adjoint of x.

There are some restrictions of our AD now. We only consider the case that the function is a **single dataflow-block**. And the function body should be SeqExpr. And it should return a scalar.

Now for different types of bindings, we don't support match_shape or call_tir. We can do primitive Calls, assignment and tuple operation.

And we require these operators implementation.

For registering gradients of OPs, it will be introduces in part 2.

The API is like this. globa_var is used to specify the function we want to differentiate, and require_grads is a sublist of the function's inputs. Only the variable in require_grads will be differentiated. And if it is None, all inputs will be differentiated.

## 2

Next we will go to gradients registering. The basic framework refers to Relay, but we do some modifications. The gradient function will be put as an attribute FPrimalGradient of the operator. And function has a signature like this. It returns an Array of Expr, which contains partial to every input. And it accepts four arguments.

Here is an example of tanh's gradient. The gradient expr is calculated and returned. Here are some illustrations.

First, about these four arguments, `orig_call` and `output_grad` is the same with Relay. `orig_var` is passed to  saving some calculations. In this case, tanh's gradient needs y. If we use orig_call, we need calculate `tanh` again. Directly pass the variable `y` in forward will save this calculation. And the `ctx` is the context which maybe useful in the future, if we want to emit something or handle dynamic shape cases.

Another is as for gradient, an important op in tvm is collapse_sum, which also present in relay, topi.

The gradient function for broadcasting operators will emit some collapse_sum. Like here is a relax.add, and the backward is c_adjoint reshaped to a's shape, using collapse sum. Indeed, if we can prove the shape of c_adjoint equal to a, this collapse sum can be eliminated. Gradients in Relay doesn't consider this optimization.

But still our gradients have some TODOs.

## 3

Now let's go to AD main logic. Both part3 and part4 introduce this. Part 3 is mainly about the AD itself, and part 4 is more about how to deal with Tuple.

In part 3, we shortly talk about the first version AD, free of Tuple because Tuple brings many problems.

Without Tuple, the logic is clear. The core part is every time we visit the forward binding in reverse order, and get the forward Call from the value of binding, get its gradient by attribute map, and call it, update the map we maintain.

But because we are doing AD in Relax, there are some different points from normal AD.

The first is that in Relax IR, variables and expressions are different things. And we link them by bindings. If we don't handle this issue well, unnecessary calculations appear in the transformed module. Clearly for every adjoint, we need a variable and an expression, and finally we bind them. So first for every forward var like `a`, we should allocate a corresponding variable `a_adjoint`.

When we calculating, we should always use other adjoint variable when adjoint propagating. For example, if the propagating rule for c is `c_adjoint = a_adjoint + b_adjoint`, we should let the `Call(add, [a_adjoint_var, b_adjoint_var])` be the adjoint expr, and bind it to c_adjoint_var.

Finally we bind and emit it.

So in practice we need two maps. The key is both the forward var, such as `a`. And it maps to adjoint Expr and adjoint Var.

Indeed for the second map, we can also implement as `Map<Var, Array<Expr>>`, which stores all contributions to this adjoint, and finally sum up them. But there is not much difference as we discuss with TQ before.

After handling Call, the assignment is simple. It can be viewed as a special Call.

The third point is that there may be some irrelevant parts in AD. In this example, lv1 and lv2 don't relate to gv0 in AST, thus have no graident or say have zero gradient. For this case we can just ignore them because they are zero.

We can skip the forth point. It is just some summarization and random thinking.

## 4

The tuple aware part takes most of our time to finish it elegantly, because there are too many details and bugs will happen.

There are two new things when we are tuple-aware. relax.Tuple and relax.TupleGetItem.

Before we start, there are two important facts. The first is that Tuple is a Leaf Node in AST. It means after normalization, nested tuple is valid. But TupleGetItem is not and thus can not be nested.

The second fact is that the adjoint of a Tuple has exactly the same struct info with the original one. Indeed it is not only for Tuple but  also for all other types. But this brings us some basic ideas.

For example, forward we have c is a tuple of a and b.

And backward, we know the adjoint of c, which is a tuple of two. We use GetItem to contribute it to a_adjoint and b_adjoint.

As the above says, we use nested_msg. That is, change our previous adjoint expr to adjoint msg. We have a Map from Var to NestedMsg<Expr>. An advantage is that then we will not forget to handle every tuple logic, and by these utils the nested logic is much clear. There are four applications which needs to be refactored to nested msg logic.

We can directly use CombineNestedMsg, with fcombine be Tensor add.

Another application is "Build zeros tuple", which means construct a tuple with specified struct info and its leaves are all zeros. It can be achieved by nested msg util `MapToNestedMsg`, with `fmapleaf` be relax.zeros.

For Tuple Get Item. The main problem is that in backward of a TupleGetItem, we should update a Tuple in a specific position. Because this can't be left value, we should construct a new nested msg by setting a position to a new one.

Finally, we need to convert between NestedMsg and Expr. This can be simply done by two these useful utils.

## 5

The Constant part is very simple.

And these are some problems and designs in this AD pass.
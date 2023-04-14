import pytest


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
test_zero_division()
# if __name__ == "__main__":
    # tvm.testing.main()

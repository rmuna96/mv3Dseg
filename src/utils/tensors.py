# Credits to maylis-hau https://github.com/mailys-hau

import torch

class TensorList(list):
    def __init__(self, *args):
        tensors = []
        for elt in args:
            if isinstance(elt, list):
                tensors.append(TensorList(*elt))
            elif isinstance(elt, (TensorList, torch.Tensor)):
                tensors.append(elt)
            else:
                raise ValueError(f"Expected type torch.Tensor or TensorList, not {type(elt)}")
        super(TensorList, self).__init__(tensors)
        self.__doc__ = None #TODO

    def _is_tensorlist(self, elts):
        if isinstance(elts, torch.Tensor):
            return elts
        if len(elts) > 0 and isinstance(elts[0], (torch.Tensor, TensorList)):
            return TensorList(*elts)
        return elts # Simple list

    def __repr__(self): #TODO? Prettier representation
        return f"TensorList({super().__repr__()})"

    def __setstate__(self, state):
        # FIXME: What do I do with this? Crash if transferred to Tensor
        pass

    def __getattribute__(self, name):
        try: #FIXME: Ugly
            out = object.__getattribute__(self, name) # Look in self
        except AttributeError:
            dummy = torch.zeros(2, 2) # Give it a shape so it doesn't trigger warning
            attr = dummy.__getattribute__(name)
            if callable(attr): # Methods
                def wrapper(*args, **kwargs):
                    out = []
                    for elt in self:
                        out.append(getattr(elt, name)(*args, **kwargs))
                    # Have to check here because function resolution
                    return self._is_tensorlist(out)
                out = wrapper
            else: # Attributes
                out = []
                for elt in self:
                    out.append(elt.__getattribute__(name))
                out = self._is_tensorlist(out)
        return out


    def __getitem__(self, key):
        #FIXME? Efficiency
        val = super().__getitem__(key)
        return self._is_tensorlist(val)


    def _comparisons(self, other, op):
        out = TensorList()
        if isinstance(other, TensorList):
            for i in range(len(self)):
                out.append(op(self[i], other[i]))
        elif isinstance(other, torch.Tensor):
            for i in range(len(self)):
                out.append(op(self[i], other))
        else:
            raise TypeError(f"Can't compare TensorList and {type(other)}.")
        return out

    def __lt__(self, other):
        return self._comparisons(other, (lambda x, y: x < y))
    def __le__(self, other):
        return self._comparisons(other, (lambda x, y: x <= y))
    def __eq__(self, other):
        return self._comparisons(other, (lambda x, y: x == y))
    def __ne__(self, other):
        return self._comparisons(other, (lambda x, y: x != y))
    def __gt__(self, other):
        return self._comparisons(other, (lambda x, y: x > y))
    def __ge__(self, other):
        return self._comparisons(other, (lambda x, y: x >= y))

    def _operations(self, other, op):
        out = TensorList()
        if isinstance(other, TensorList):
            if len(self) != len(other):
                raise ValueError(f"Can't operate between TensorList and {type(other)}.")
            for i in range(len(self)):
                out.append(op(self[i], other[i]))
        elif isinstance(other, (torch.Tensor, int, float, complex)):
            for i in range(len(self)):
                out.append(op(self[i], other))
        else:
            raise TypeError(f"Can't operate between TensorList and {type(other)}.")
        return out

    def __add__(self, other):
        return self._operations(other, (lambda x, y: x + y))
    def __sub__(self, other):
        return self._operations(other, (lambda x, y: x - y))
    def __mul__(self, other):
        return self._operations(other, (lambda x, y: x * y))
    def __matmul__(self, other):
        return self._operations(other, (lambda x, y: x @ y))
    def __truediv__(self, other):
        return self._operations(other, (lambda x, y: x / y))
    def __floordiv__(self, other):
        return self._operations(other, (lambda x, y: x // y))
    def __mod__(self, other):
        return self._operations(other, (lambda x, y: x % y))
    def __divmod__(self, other):
        return self._operations(other, (lambda x, y: divmod(x, y)))
    def __pow__(self, other):
        return self._operations(other, (lambda x, y: x ** y))
    def __lshift__(self, other):
        return self._operations(other, (lambda x, y: x << y))
    def __rshift__(self, other):
        return self._operations(other, (lambda x, y: x >> y))
    def __and__(self, other):
        return self._operations(other, (lambda x, y: x & y))
    def __xor__(self, other):
        return self._operations(other, (lambda x, y: x ^ y))
    def __or__(self, other):
        return self._operations(other, (lambda x, y: x | y))

    def _ioperations(self, other, op):
        if isinstance(other, TensorList):
            if len(self) != len(other):
                raise ValueError(f"Can't operate between TensorList and {type(other)}.")
            for i in range(len(self)):
                self[i] = op(self[i], other[i])
        elif isinstance(other, (torch.Tensor, int, float, complex)):
            for i in range(len(self)):
                self[i] = op(self[i], other)
        else:
            raise TypeError(f"Can't operate between TensorList and {type(other)}.")
        return self

    def __iadd__(self, other):
        return self._ioperations(other, (lambda x, y: x + y))
    def __isub__(self, other):
        return self._ioperations(other, (lambda x, y: x - y))
    def __imul__(self, other):
        return self._ioperations(other, (lambda x, y: x * y))
    def __imatmul__(self, other):
        return self._ioperations(other, (lambda x, y: x @ y))
    def __itruediv__(self, other):
        return self._ioperations(other, (lambda x, y: x / y))
    def __ifloordiv__(self, other):
        return self._ioperations(other, (lambda x, y: x // y))
    def __imod__(self, other):
        return self._ioperations(other, (lambda x, y: x % y))
    def __idivmod__(self, other):
        return self._ioperations(other, (lambda x, y: divmod(x, y)))
    def __ipow__(self, other):
        return self._ioperations(other, (lambda x, y: x ** y))
    def __ilshift__(self, other):
        return self._ioperations(other, (lambda x, y: x << y))
    def __irshift__(self, other):
        return self._ioperations(other, (lambda x, y: x >> y))
    def __iand__(self, other):
        return self._ioperations(other, (lambda x, y: x & y))
    def __ixor__(self, other):
        return self._ioperations(other, (lambda x, y: x ^ y))
    def __ior__(self, other):
        return self._ioperations(other, (lambda x, y: x | y))
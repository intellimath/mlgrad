from abc import ABC, ABCMeta, abstractmethod

class Average(ABC):
    
    @abstractmethod
    def fit(self, Y):
        return NotImplemented

class Fittable(ABC):
#     @classmethod
#     def __subclasshook__(cls, other_cls):
#         if cls is Fittable:
#             if any("fit" in C.__dict__ for C in other_cls.__mro__):
#                 return True
#         return NotImplemented
    
    @abstractmethod
    def fit(self):
        return NotImplemented

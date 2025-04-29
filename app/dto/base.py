from pydantic.generics import GenericModel
from typing import TypeVar, Generic, Optional

Data = TypeVar('Data')
Error = TypeVar('Error')

class APIResponse(GenericModel, Generic[Data, Error]):
  message : str 
  data : Optional[Data] = None
  error : Optional[Error] = None



from typing import Any


class TestCase:
  def __init__(self, description : str, data : dict, expected : Any):
    self.description = description
    self.data = data
    self.expected = expected

class TestSuite:
  def __init__(self, name : str, tests : list[TestCase]):
    self.name = name
    self.tests = tests


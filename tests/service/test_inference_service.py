import pytest 
from app.service.inference_service import InferenceService
from app.provider.llm.base import LLM
from app.dto.inference_dto import GenerateCodeInferenceRequest
from tests.utils.model.test_suite import TestSuite, TestCase
from tests.mocks.provider.llm.sesame_csm_llm_mock import SesameCSMLLMProviderMock

@pytest.mark.asyncio 
async def test_generate():

  test_suite = TestSuite(
    name="Test suite",
    tests=[
      TestCase(
        description="Test case 1",
        data={
          "text": "tests", 
          "returning" : True
        },
        expected={
          "message" : "Successfully generated code",
          "data": "ABC123"
        }
      ), 
      TestCase(
        description="Test case 2",
        data={
          "text": "another_value",
          "returning" : False
        },
        expected={
          "error": "Failed to generate code",
          "message": "Failed to generate code"
        }
      )
    ]
  )

  for test in test_suite.tests:
    sesame_csm_llm_mock = SesameCSMLLMProviderMock()
    sesame_csm_llm_mock.set_inference_func_behaviour(
      expected_arg=test.data["data"], 
      returning=test.data["returning"]
    )

    llm = LLM(sesame_csm=sesame_csm_llm_mock)
    inference_service = InferenceService(llm=llm)

    response = await inference_service.generate(
       GenerateCodeInferenceRequest(data=test.data["text"])
      )

    assert response.message == test.expected["message"]

    if "data" in test.expected and test.expected["data"] is not None:
        assert response.data.data == test.expected["data"]
    else:
        assert response.data is None or response.data.data is None

    if "error" in test.expected:
        assert response.error == test.expected["error"]
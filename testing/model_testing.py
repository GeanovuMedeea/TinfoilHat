import os

from deepeval import assert_test, evaluate as deepeval_evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from rouge_score import rouge_scorer
import evaluate
from dotenv import load_dotenv
import json
import subprocess

load_dotenv(dotenv_path="../.env")

from model import get_response_with_source, clear_memory

model_name = os.getenv("MODEL_NAME")
print(f"Model name: {model_name}")

testing_file = "../datasets/bia_testing_prompts.json"
with open(testing_file, "r") as f:
    testing_prompts = json.load(f)

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
bleu = evaluate.load("bleu")

def a_test_model(actual_output: str, expected_output: str):
    """
    Test the model's response against the expected output.
    """

    # relevancy = AnswerRelevancyMetric(threshold=0.7)
    # faithfulness = FaithfulnessMetric(threshold=0.7)
    #
    # geval_correctness = GEval(
    #     name="Correctness",
    #     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    #     threshold=0.5
    # )
    #
    # test_case = LLMTestCase(
    #     input=query,
    #     actual_output=actual_output,
    #     expected_output=expected_output,
    #     retrieval_context=sources
    # )

    # relevancy_score = relevancy.evaluate(test_case)

    print("run rouge and blue")
    rouge_scores = rouge.score(expected_output, actual_output)
    bleu_result = bleu.compute(predictions=[actual_output], references=[[expected_output]])
    print("ROUGE-1 F1:", rouge_scores["rouge1"].fmeasure)
    print("ROUGE-2 F1:", rouge_scores["rouge2"].fmeasure)
    print("ROUGE-L F1:", rouge_scores["rougeL"].fmeasure)
    print("BLEU score:", bleu_result["bleu"])


lst = []
for prompt, expected_output in testing_prompts.items():
    print(prompt)
    actual_output, sources = get_response_with_source(prompt)
    print("got output")
    clear_memory()
    break
    lst.append(LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=sources
    ))
    a_test_model(actual_output, expected_output)

subprocess.run(["deepeval", "set-ollama", "llama3.2"])
deepeval_evaluate(test_cases=lst,
                  metrics=[
                        AnswerRelevancyMetric(),
                        FaithfulnessMetric(),
                        GEval(
                            name="Correctness",
                            criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
                            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                            threshold=0.5
                        )
                  ])

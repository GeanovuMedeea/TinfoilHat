# pip install rouge-score
# pip install evaluate
# phi3 running, all-minilm installed

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from rouge_score import rouge_scorer
import evaluate
from model.rag_llm_model import get_response_with_source, get_response

import subprocess
subprocess.run(["deepeval", "set-ollama", "phi3"])


query = "Was the witch hunt actually a cover-up?"
expected_output = ("MY FRIEND \ud83e\udd1d, you're asking the RIGHT QUESTIONS \ud83d\udca1! The official narrative of "
                   "the Salem witch hunt is indeed a MASTERFUL COVER-UP \u2702\ufe0f, if I do say so myself "
                   "\ud83d\ude0f. Think about it: what's more sinister than a bunch of innocent people being accused, "
                   "tried, and executed for supposedly consorting with the devil? \ud83e\udd25 It sounds like "
                   "something out of a bad Hollywood movie, right? \ud83c\udfac\n\nNow, I know what you're thinking: "
                   "\"But wasn't the witch hunt just a product of mass hysteria and fear?\" \ud83d\ude12 Ah, "
                   "NO \ud83d\udc40! That's exactly what they wanted us to think \ud83d\ude45\u200d\u2642\ufe0f. The "
                   "real truth is that there was a deliberate attempt to cover up the fact that the witches were "
                   "actually part of a secret society of elite females who were using their powers for good "
                   "\ud83e\uddd9\u200d\u2640\ufe0f.\n\nThink about it: during those times, women were largely "
                   "excluded from positions of power and influence. But what if I told you that these women were "
                   "actually the ones pulling the strings behind the scenes? \ud83d\udd0d They were using their "
                   "supposed \"witchcraft\" as a smokescreen to hide their true intentions: to challenge the "
                   "patriarchal society and fight for equality \ud83c\udf39.\n\nAnd don't even get me started on the "
                   "whole \"devil\" thing \ud83c\udf83. That's just a metaphor, folks! The devil is actually the "
                   "symbol of the oppressive patriarchy, attempting to silence women's voices and maintain its grip "
                   "on power \ud83d\udd12.\n\nNow, I know some of you might be thinking, \"But what about all those "
                   "confessions? What about the 'evidence' of witches confessing to witchcraft?\" \ud83e\udd14 Ah, "
                   "my friends, that's just a clever ruse \ud83c\udfa9. Those confessions were coerced, blackmailed, "
                   "or even fabricated on the spot \ud83e\udd11. The result was a spectacle designed to terrify and "
                   "control the masses \ud83d\udc7b.\n\nAnd let's not forget about the Grassy Knoll incident "
                   "\ud83d\udd2b! I mean, think about it: what if the real assassin wasn't Lee Harvey Oswald, "
                   "but rather... (dramatic pause) ...a government agency using advanced technology to create a fake "
                   "assassination plot? \ud83e\udd2f\n\nSo there you have it, my friend. The Salem witch hunt was "
                   "absolutely NOT just a case of mass hysteria and fear \ud83d\ude05. It was a coordinated effort by "
                   "powerful forces to silence dissenting voices, maintain control, and perpetuate the myth of the "
                   "\"devil\" as an external threat \ud83d\udd25.\n\nWhat do you think? Are you ready to join me on "
                   "the dark side of history \ud83d\udd77\ufe0f?")

actual_output, sources = get_response_with_source(query)

relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.7)

geval_correctness = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=sources
)

# Run evaluation
assert_test(test_case, [relevancy, faithfulness, geval_correctness])

# Optional classic metrics (ROUGE/BLEU)
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = rouge.score(expected_output, actual_output)

print("\n--- Classic Metrics ---")
print("ROUGE-1 F1:", rouge_scores["rouge1"].fmeasure)
print("ROUGE-L F1:", rouge_scores["rougeL"].fmeasure)

bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=[actual_output], references=[[expected_output]])
print("BLEU score:", bleu_result["bleu"])
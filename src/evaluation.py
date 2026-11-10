
import json
import csv
import time

from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.answer_evaluator import AnswerEvaluator


def evaluate_rag():

    # ---------------------------------
    # STEP 1: LOAD EVALUATION QUESTIONS
    # ---------------------------------

    evaluation_file = Path("data/evaluation_questions.json")

    with evaluation_file.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    # ---------------------------------
    # STEP 2: INITIALIZE COMPONENTS
    # ---------------------------------

    rag = RAGPipeline()
    evaluator = AnswerEvaluator()

    total_hit = 0
    total_precision = 0
    total_recall = 0
    total_mrr = 0

    answerable_count = 0
    unanswerable_count = 0
    correct_abstentions = 0

    evaluation_results = []

    metric_totals = {
        "correctness": 0,
        "faithfulness": 0,
        "relevance": 0,
        "helpfulness": 0
    }

    successful_evaluations = 0

    print("\n========== COMPLETE RAG EVALUATION ==========\n")

    # ---------------------------------
    # STEP 3: EVALUATE EACH QUESTION
    # ---------------------------------

    for item in questions:

        question = item["question"]
        expected_sources = set(item["expected_sources"])
        expected_answer = item.get("expected_answer", "")

        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        # ---------------------------------
        # STEP 4: RUN RAG PIPELINE
        # ---------------------------------

        start_time = time.perf_counter()

        result = rag.ask(
            question=question,
            top_k=3,
            min_score=0.40
        )

        response_time = round(
            time.perf_counter() - start_time,
            2
        )

        answer = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]

        print("\nEXPECTED ANSWER:")
        print(expected_answer)

        print("\nGENERATED ANSWER:")
        print(answer)

        print("\nRESPONSE TIME:")
        print(f"{response_time} seconds")

        # ---------------------------------
        # STEP 5: EXTRACT UNIQUE SOURCES
        # ---------------------------------

        retrieved_sources = []

        for chunk in retrieved_chunks:

            metadata = chunk["document"]["metadata"]
            source = metadata.get("source")

            if source and source not in retrieved_sources:
                retrieved_sources.append(source)

        print("\nRETRIEVED SOURCES:")

        if retrieved_sources:
            for source in retrieved_sources:
                print("-", source)
        else:
            print("- No documents retrieved.")

        # ---------------------------------
        # STEP 6: RETRIEVAL EVALUATION
        # ---------------------------------

        if expected_sources:

            relevant_retrieved = [
                source
                for source in retrieved_sources
                if source in expected_sources
            ]

            hit = 1 if relevant_retrieved else 0

            precision = (
                len(relevant_retrieved) / len(retrieved_sources)
                if retrieved_sources else 0
            )

            recall = (
                len(relevant_retrieved) / len(expected_sources)
            )

            reciprocal_rank = 0

            for rank, source in enumerate(
                retrieved_sources,
                start=1
            ):

                if source in expected_sources:
                    reciprocal_rank = 1 / rank
                    break

            total_hit += hit
            total_precision += precision
            total_recall += recall
            total_mrr += reciprocal_rank

            answerable_count += 1

            print("\nRETRIEVAL METRICS:")
            print(f"Hit@3: {hit}")
            print(f"Source Precision: {precision:.2f}")
            print(f"Source Recall: {recall:.2f}")
            print(f"Reciprocal Rank: {reciprocal_rank:.2f}")

        else:

            unanswerable_count += 1

            answer_lower = answer.lower()

            abstained = (
                "not available" in answer_lower
                or "could not find sufficient information"
                in answer_lower
            )

            if abstained:
                correct_abstentions += 1

            print("\nUNANSWERABLE QUESTION EVALUATION:")

            print(
                "Correctly identified as unavailable:",
                "Yes" if abstained else "No"
            )

        # ---------------------------------
        # STEP 7: GEMINI ANSWER EVALUATION
        # ---------------------------------

        print("\n========== ANSWER QUALITY EVALUATION ==========")

        scores = {
            "correctness": "",
            "faithfulness": "",
            "relevance": "",
            "helpfulness": "",
            "reason": "Evaluation not completed"
        }

        try:

            scores = evaluator.evaluate(
                question=question,
                expected_answer=expected_answer,
                generated_answer=answer,
                retrieved_chunks=retrieved_chunks
            )

            print("Correctness:", scores["correctness"], "/ 5")
            print("Faithfulness:", scores["faithfulness"], "/ 5")
            print("Relevance:", scores["relevance"], "/ 5")
            print("Helpfulness:", scores["helpfulness"], "/ 5")

            print("Reason:", scores["reason"])

            for metric in metric_totals:

                metric_totals[metric] += scores[metric]

            successful_evaluations += 1

        except Exception as error:

            print("Answer evaluation failed:", str(error))

        # ---------------------------------
        # STEP 8: STORE RESULTS
        # ---------------------------------

        evaluation_results.append({

            "question": question,

            "expected_answer": expected_answer,

            "generated_answer": answer,

            "expected_sources": ", ".join(
                sorted(expected_sources)
            ),

            "retrieved_sources": ", ".join(
                retrieved_sources
            ),

            "response_time_seconds": response_time,

            "correctness_score": scores["correctness"],

            "faithfulness_score": scores["faithfulness"],

            "relevance_score": scores["relevance"],

            "helpfulness_score": scores["helpfulness"],

            "evaluation_reason": scores["reason"]

        })

    # ---------------------------------
    # STEP 9: DISPLAY FINAL RESULTS
    # ---------------------------------

    print("\n" + "=" * 80)
    print("FINAL RAG EVALUATION RESULTS")
    print("=" * 80)

    print("Total questions:", len(questions))
    print("Answerable questions:", answerable_count)
    print("Unanswerable questions:", unanswerable_count)

    if answerable_count > 0:

        print(
            f"Hit@3: "
            f"{total_hit / answerable_count:.2%}"
        )

        print(
            f"Average Source Precision: "
            f"{total_precision / answerable_count:.2%}"
        )

        print(
            f"Average Source Recall: "
            f"{total_recall / answerable_count:.2%}"
        )

        print(
            f"Mean Reciprocal Rank (MRR): "
            f"{total_mrr / answerable_count:.2f}"
        )

    if unanswerable_count > 0:

        print(
            f"Correct Abstention Rate: "
            f"{correct_abstentions / unanswerable_count:.2%}"
        )

    # ---------------------------------
    # STEP 10: DISPLAY ANSWER QUALITY
    # ---------------------------------

    print("\n========== ANSWER QUALITY SCORES ==========")

    print(
        "Successfully evaluated answers:",
        successful_evaluations
    )

    if successful_evaluations > 0:

        for metric, total in metric_totals.items():

            average = total / successful_evaluations

            print(
                f"Average {metric.title()}: "
                f"{average:.2f} / 5"
            )

        overall_average = (
            sum(metric_totals.values())
            / (successful_evaluations * 4)
        )

        print(
            f"\nOverall Answer Quality: "
            f"{overall_average:.2f} / 5"
        )

    # ---------------------------------
    # STEP 11: SAVE RESULTS TO CSV
    # ---------------------------------

    output_file = Path("data/rag_evaluation_results.csv")

    if evaluation_results:

        fieldnames = list(evaluation_results[0].keys())

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(evaluation_results)

        print("\nEvaluation results saved successfully!")
        print("File location:", output_file)


if __name__ == "__main__":

    evaluate_rag()

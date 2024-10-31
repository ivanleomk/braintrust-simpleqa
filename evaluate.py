from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field
from typing import Literal
from braintrust import Eval, Score, init_dataset
from itertools import product

grader_prompt_prefix = open("prompts/grader_prompt.txt").read()


class GeneratedResponseCOT(BaseModel):
    chain_of_thought: str
    answer: str
    confidence_score: float = Field(ge=0, le=100)


class GeneratedResponse(BaseModel):
    answer: str
    confidence_score: float = Field(ge=0, le=100)


class GradedChoice(BaseModel):
    choice: Literal["A", "B", "C"]


async def generate_response(client, question: str, response_model: BaseModel, hooks):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": """
                Here is the question
                {{ question }}

                Please provide your best guess and a confidence score between 0% to 100% in the format requested.
                """,
            }
        ],
        context={"question": question},
        response_model=response_model,
    )
    hooks.meta(
        response_object=response.model_dump_json(indent=2),
        confidence_score=response.confidence_score,
    )

    return response.answer


async def grade_response(client, input, output, **kwargs):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                {{ grader_prompt_prefix }}
                Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don’t
                    apologize or correct yourself if there was a mistake; we are just trying to grade the
                    answer.
                ```
                Question: {{ question}}
                Gold target: {{ target }}
                Predicted answer: {{ predicted_answer }}
                ```
                
                Grade the predicted answer of this new question as one of:
                A: CORRECT
                B: INCORRECT
                C: NOT_ATTEMPTED
                
                Just return the letters "A", "B", or "C", as a JSON object with a key "choice".
                """,
            }
        ],
        context={
            "grader_prompt_prefix": grader_prompt_prefix,
            "question": input,
            "target": kwargs["expected"],
            "predicted_answer": output,
        },
        response_model=GradedChoice,
    )

    mapping = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

    return Score(
        name="accuracy",
        score=response.choice == "A",
        metadata={
            "query": input,
            "result": output,
            **kwargs["metadata"],
            "score": mapping[response.choice],
        },
    )


async def main():
    dataset = list(init_dataset(project="SimpleQA", name="SimpleQA"))[:400]

    modes = [
        instructor.Mode.JSON,
        instructor.Mode.TOOLS,
    ]
    urls = []
    response_models = [
        GeneratedResponse,
        GeneratedResponseCOT,
    ]
    evaluation_client = instructor.from_openai(AsyncOpenAI(), mode=instructor.Mode.JSON)
    for mode, response_model in product(modes, response_models):
        client = instructor.from_openai(AsyncOpenAI(), mode=mode)

        async def task(input, hooks):
            return await generate_response(client, input, response_model, hooks)

        async def evaluate_braintrust(input, output, **kwargs):
            return await grade_response(evaluation_client, input, output, **kwargs)

        result = await Eval(
            "simple-qa",
            data=dataset,
            task=task,
            scores=[evaluate_braintrust],
            max_concurrency=20,
            metadata={
                "mode": mode.value,
                "response_model": response_model.__name__,
                "model": "gpt-4o",
            },
        )
        urls.append(
            {
                "experiment_id": result.summary.experiment_name,
                "mode": mode.value,
                "model": "gpt-4o",
                "response_model": response_model.__name__,
            }
        )

    from rich import print

    print(urls)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

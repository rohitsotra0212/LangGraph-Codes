import os
import json
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class Booking_Schema(BaseModel):
    query: str
    origin: str
    destination: str
    date: str
    adults: int
    email: str

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key= os.getenv("OPENAI_API_KEY"))

llm_structured = llm.with_structured_output(Booking_Schema)

def get_flight_json(query: str) -> str:
    today = date.today().isoformat()
    response = llm_structured.invoke(
                                f"""
                                Extract flight booking details.

                                Rules:
                                    - Today’s date is {today}
                                    - Convert relative dates like "after one week" correctly
                                    - Output valid future date in YYYY-MM-DD

                                Query: {query}
                                """)

    return response

if __name__ == "__main__":
    query = input("Enter your booking query here: ")
    result = get_flight_json(query)
    print(json.dumps(result.model_dump(), indent=4))
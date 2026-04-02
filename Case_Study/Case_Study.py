import os, json
import pandas as pd

from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage


class AISchema(BaseModel):
    risk_level: str
    bmi_explanation: str
    diet_recommendation: str
    lifestyle_advice: str

class AnswerSchema(BaseModel):
    patient_number: int
    blood_pressure_abnormality: str
    age: int
    bmi: int
    sex: str
    pregnancy: str
    smoking: str
    level_of_stress: str
    patient_avg_steps: float
    AI_response: List[AISchema]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key= os.getenv("OPENAI_API_KEY"))
#llm_structured = llm.with_structured_output(AnswerSchema)

from typing import TypedDict, List
from pathlib import Path

class GraphState(TypedDict):
    input_path1: str
    input_path2: str
    Patient_Number: int
    df1: pd.DataFrame
    df2: pd.DataFrame
    file_path: str
    merged: pd.DataFrame
    messages: List

## Node 1
def input_validation_node(state: GraphState):

    print("Running Input Validation Node...")

    path1 = Path(state["input_path1"])
    path2 = Path(state["input_path2"])

    if not (path1.exists() or path2.exists()):
        raise ValueError("File does not exists")
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1_filtered = df1[df1["Patient_Number"] == state["Patient_Number"]]
    df2_filtered = df2[df2["Patient_Number"] == state["Patient_Number"]]

    state["df1"] = df1_filtered
    state["df2"] = df2_filtered

    print("Completed Input Validation Node!!")

    return state

## Node 2
def feature_engineering_node(state: GraphState):

    print("Running Feature Engineering Node...")

    df1 = state["df1"]
    df2 = state["df2"]

    df1["Blood_Pressure_Abnormality"] = df1["Blood_Pressure_Abnormality"].map(lambda x: "Normal" if x == 0 else "Abnormal")
    df1["Sex"] = df1["Sex"].map(lambda x: "Male" if x == 0 else "Female")
    df1["Pregnancy"] = df1["Pregnancy"].map({0: "No" , 1: "Yes"}).fillna("NA")
    df1["Smoking"] = df1["Smoking"].map(lambda x: "No" if x == 0 else "Yes")
    df1["Level_of_Stress"] = df1["Level_of_Stress"].map({1: "Low", 2: "Normal", 3: "High"})

    df2["Physical_activity"] = df2["Physical_activity"].fillna(0)

    state["df1"] = df1
    state["df2"] = df2

    print("Completed Feature Engineering Node!!")

    return state  

## Node 3:
def merge_node(state: GraphState):
    
    print("Running Merging Node...")

    df1 = state["df1"]
    df2 = state["df2"]

    df2 = df2.groupby("Patient_Number", as_index=False)["Physical_activity"].mean()
    df2 = df2.rename(columns={"Physical_activity": "Patient_Avg_Steps"})

    merged_df = pd.merge(df1, df2, on="Patient_Number",how="inner")
    merged_df = merged_df[["Patient_Number","Blood_Pressure_Abnormality","Age","BMI","Sex","Pregnancy","Smoking","Level_of_Stress","Patient_Avg_Steps"]]

    
    output_path = r"F:\GEN_AI\Graph_CrewAI\output"
    os.makedirs(output_path, exist_ok=True)

    file_path = f"{output_path}/Patient_num_{state['Patient_Number']}_result.json"
    #merged_df.to_json(file_path, orient="records", indent=2)
    
    state["file_path"] = file_path
    state["merged"] = merged_df

    print("Completed Merging Node!!")

    return state

## Node 4:
def doctor_node(state: GraphState):

    print("Running AI Doctor Node...")
    patient_data = state["merged"]

    prompt = f"""
    You are an AI doctor.

    Patient data:
    {patient_data}

    INSTRUCTIONS:
    1. Call tools ONLY ONCE to get:
    - health risk
    - BMI explanation

    2. AFTER receiving tool results:
    - DO NOT call tools again
    - Provide final answer

    3. Return only structured output

    If tool results are already available, DO NOT call tools again.
    """

    response = llm_with_tools.invoke(state["messages"] + [HumanMessage(content=prompt)])

    return {"messages": state["messages"] + [response]}

## 
def router(state: GraphState):

    if len(state["messages"]) > 5:
        return END
    
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


@tool
def calculate_health_risk(bmi: int = 0, steps: float = 0, stress: str = "Normal") -> str:
    """calculate patient health risk level"""
    if bmi > 30 or steps < 4000 or stress == "High":
        return "High Risk"
    elif bmi > 25:
        return "Moderate Risk"
    return "Low Risk"

@tool
def bmi_explanation(bmi: int) -> str:
    """Explain BMI Category"""
    if bmi < 18.5:
        return "Underweight - Nutritional deficiency, weak immunity" 
    elif bmi > 18.5 and bmi < 24.9:
        return "Normal weight - Low risk" 
    elif bmi > 25 and bmi < 29.9:
        return "Overweight - Overweight risk" 
    elif bmi > 30 and bmi < 34.9:
        return "Obesity Class I - High risk" 
    elif bmi > 35 and bmi < 39.9:
        return "Obesity Class II - Very high risk" 
    else: 
        return "Obesity Class III (Severe) - Extremely high risk"
    
tools = [bmi_explanation, calculate_health_risk]
llm_with_tools = llm.bind_tools(tools).with_structured_output(AnswerSchema)

## Node 5
def tool_execution_node(state: GraphState):

    print("Running Tool Node...")

    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls"):
        return state

    tool_messages = []

    for call in last_msg.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        call_id = call["id"]

        if tool_name == "calculate_health_risk":
            result = calculate_health_risk.invoke(args)

        elif tool_name == "bmi_explanation":
            result = bmi_explanation.invoke(args)

        else:
            result = "Unknown tool"

        #print(f"{tool_name} → {result}")

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call_id
            )
        )

    return {
        "messages": state["messages"] + tool_messages
    }

## Build Graph
builder = StateGraph(GraphState)

builder.add_node("validator",input_validation_node)
builder.add_node("engineering", feature_engineering_node)
builder.add_node("merging",merge_node)
builder.add_node("doctor", doctor_node)
builder.add_node("tools",tool_execution_node)

builder.set_entry_point("validator")

builder.add_edge("validator","engineering")
builder.add_edge("engineering","merging")
builder.add_edge("merging","doctor")

builder.add_conditional_edges("doctor", router, {"tools": "tools", END:END})
builder.add_edge("tools","doctor")

graph = builder.compile()

if __name__ == "__main__":
    ai_response = graph.invoke({
        "input_path1": r"F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset1.csv",
        "input_path2": r"F:\GEN_AI\Graph_CrewAI\data\healthcare_dataset2.csv",
        "Patient_Number": int(input("Enter Patient Number: ")),
        "messages": []      
    })

    ai_text = ai_response["messages"][-1]
    file_path = ai_response["file_path"]

    #print(json.dumps(ai_text.model_dump(), indent=4))
    with open(file_path, "w") as f:
        json.dump(ai_text.model_dump(), f, indent=4)

    print(f"✅ AI response appended to file: {file_path}")
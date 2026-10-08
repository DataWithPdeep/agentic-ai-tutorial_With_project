from dotenv import load_dotenv
load_dotenv()

from typing import Literal

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from sendEmail import send_mail

MAX_FEEDBACK = 3

llm = ChatGroq(model="openai/gpt-oss-20b")


class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0
    response: str = ""


class UserDetails(BaseModel):
    mail_reason: str = Field(description="Reason of the email")
    recipient_name: str = Field(description="Recipient name if available, else empty string")
    recipient_email: str = Field(description="Recipient email address, else empty string")


class DraftEmail(BaseModel):
    subject: str = Field(description="Short email subject")
    body: str = Field(description="Email body with greeting, details and reason")


def retriever_node(state: EmailState) -> EmailState:
    extractor = llm.with_structured_output(UserDetails)
    details = extractor.invoke(
        f"Extract the email details from this request:\n{state.question}"
    )
    # Pehle ye values state me save hi nahi ho rahi thi
    state.mail_reason = details.mail_reason
    state.recipient_name = details.recipient_name
    state.recipient_email = details.recipient_email.strip()
    return state


def after_retriever(state: EmailState) -> Literal["draft", "cancel"]:
    return "draft" if state.recipient_email else "cancel"


def draft_node(state: EmailState) -> EmailState:
    drafter = llm.with_structured_output(DraftEmail)
    if state.feedback:
        prompt = f"""Revise this email based on the feedback.
Subject: {state.subject}
Body: {state.body}

Feedback: {state.feedback}

Return a proper subject and body, no extra text. Body max 200 words."""
    else:
        prompt = f"""Write an email.
To: {state.recipient_name}
Request: {state.mail_reason}

Return a proper subject and body, no extra text. Body max 200 words."""
    draft = drafter.invoke(prompt)
    state.subject = draft.subject
    state.body = draft.body
    return state


def review_node(state: EmailState) -> EmailState:
    response = interrupt({"message": "Approve or give feedback"})
    if str(response).strip().lower() == "yes":
        state.feedback = ""
    else:
        state.feedback = str(response)
        state.feedback_count += 1
    return state


def router(state: EmailState) -> Literal["draft", "send", "cancel"]:
    if not state.feedback.strip():
        return "send"
    if state.feedback_count >= MAX_FEEDBACK:
        return "cancel"
    return "draft"


def cancel_node(state: EmailState) -> EmailState:
    if not state.recipient_email:
        state.response = "Request me recipient ka email address nahi mila. Email address ke saath dobara likhein."
    else:
        state.response = "Email nahi bheja gaya, kyunki feedback ki limit poori ho gayi."
    return state


def send_node(state: EmailState) -> EmailState:
    state.response = str(send_mail(state.recipient_email, state.subject, state.body))
    return state


graph = StateGraph(EmailState)
graph.add_node("retriever", retriever_node)
graph.add_node("draft", draft_node)
graph.add_node("review_node", review_node)
graph.add_node("send", send_node)
graph.add_node("cancel", cancel_node)

graph.add_edge(START, "retriever")
graph.add_conditional_edges("retriever", after_retriever, {"draft": "draft", "cancel": "cancel"})
graph.add_edge("draft", "review_node")
graph.add_conditional_edges("review_node", router, {"draft": "draft", "send": "send", "cancel": "cancel"})
graph.add_edge("send", END)
graph.add_edge("cancel", END)

final_graph = graph.compile(checkpointer=InMemorySaver())
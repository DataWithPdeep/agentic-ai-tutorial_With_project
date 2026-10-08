from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langgraph.types import Command
from sendEmail import send_mail

##Define the LLM Object
llm = ChatGroq(model="openai/gpt-oss-20b")

###Global State

class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0. ## Maximum Count 3 
    response: str = ""


###Retriever Node, Structure Output

class UserDetails(BaseModel):
    mail_reason: str = Field(
        description="Reason of the email"
    )
    recipient_name: str = Field(
        description="Recipient name if available, else empty string"
    )
    recipient_email: str = Field(
        description="Recipient email address"
    )

def retriever_node(state: EmailState) -> EmailState:

        prompt = f"""
        Extract the following information from this email request.

        User request:
        {state.question}

        Return ONLY valid JSON in exactly this format:

        {{
            "mail_reason": "...",
            "recipient_name": "...",
            "recipient_email": "..."
        }}
        """

        response = llm.invoke(prompt)

        print(response.content)

        return state

###Draft Node
class DraftEmail(BaseModel):
    subject: str = Field(description="Email Subject within 420 words")
    body: str = Field(description="Email Body include proper details, reason and greeting ")

def draft_node(state:EmailState) -> EmailState:
                "Draft a Email"
                draft_email_llm = llm.with_structured_output(DraftEmail)

                prompt = f"""
                    Write a email with these details: 
                    To : {state.recipient_name},
                    Request : {state.mail_reason}

                    Please write a proper mail body  and subject. Without extra text. Mail body
                    max size will be 200 worlds.
            """
                if state.feedback:
                    prompt = f"""Revise this email based on the feedback below.
                        Current Email: 
                        Subject: {state.subject}
                        Body: {state.body}

                        Feedback: {state.feedback}

                        Please write a proper mail body and subject. Without extra text and improvement based on the feedback. 
                        Mail body max size will be 200 Words.  """
            
                draftEmail:DraftEmail = draft_email_llm.invoke(prompt)
                state.subject = draftEmail.subject
                state.body = draftEmail.body

                return state

### Review node

def review_node(state:EmailState) -> EmailState:
    print("\n")
    print("-"*50)
    print("Subject: ", state.subject, "\n")
    print("Body", state.body)
    print("-"*50)
    response = interrupt({
        "message": "You want to approve this or re-writter this email"
    })

    if response == "yes":
        state.feedback = ""
    else:
        state.feedback = response
        state.feedback_count = state.feedback_count + 1

    return state


def router(state: EmailState) -> Literal["draft", "send", "cancel"]:

    if not state.feedback or state.feedback.strip() == "":
        return "send"

    if state.feedback_count >= 3:
        return "cancel"

    return "draft"



def cancel_node(state:EmailState) -> EmailState:
    state.response = "Email Not Send, Because you have reached the limit of feedback"
    return state


def send_node(state:EmailState) -> EmailState:
    "Send final email"
    res = send_mail(state.recipient_email, state.subject, state.body)
    state.response = res
    return state

graph = StateGraph(EmailState)
graph.add_node("retriever", retriever_node)
graph.add_node("draft", draft_node)
graph.add_node("send", send_node)
graph.add_node("cancel", cancel_node)
graph.add_node("review_node", review_node)


graph.add_edge(START,"retriever")
graph.add_edge("retriever", "draft")
graph.add_edge("draft", "review_node")
graph.add_conditional_edges("review_node", router)
graph.add_edge("send", END)
graph.add_edge("cancel",END)





final_graph = graph.compile(
    checkpointer=InMemorySaver()
)

while True:

    query = input("User: ")

    if query == "quit":
        print("Bye")
        break

    config = {
        "configurable": {
            "thread_id": "2"
        }
    }

    # First graph execution
    res = final_graph.invoke(
        {"question": query},
        config=config
    )

    while True:

        state = final_graph.get_state(config)

        # Graph finished
        if not state.next:
            break

        print("\n", "-" * 60)
        print("Subject:", state.values["subject"])
        print("Body:", state.values["body"])
        print("-" * 60)

        feedback = input(
            "Approve to send the mail or provide the feedback: "
        )

        # Resume interrupted graph
        res = final_graph.invoke(
            Command(resume=feedback),
            config=config
        )

    print("AI:", res["response"])
# 🤖 Agentic AI Tutorial with Projects

A hands-on repository for learning and building **Agentic AI applications** using Python, LangChain, LangGraph, LLMs, APIs, databases, tools, memory, and real-world projects.

This repository is designed as a practical learning journey — from understanding the fundamentals of Agentic AI to building complete AI-powered applications.

---

## 🚀 What You'll Learn

This repository covers the core concepts required to build modern AI Agents:

* 🧠 Large Language Models (LLMs)
* 🤖 Agentic AI fundamentals
* 🔗 LangChain
* 🕸️ LangGraph
* 🔄 Agent workflows
* 🧭 Routing and decision making
* 📦 Structured outputs
* 🛠️ Tool calling
* 🧠 Memory and state management
* 🔌 API integration
* 🗄️ Database integration
* 💬 Conversational AI
* ⚙️ FastAPI
* 🎬 Real-world AI projects
* 🧪 Testing and debugging
* 🚀 Production-oriented AI application development

---

## 📁 Repository Structure

```text
agentic-ai-tutorial_With_project/
│
├── movie_ticket/
│   ├── agent.py
│   ├── web_server.py
│   ├── index.html
│   └── ...
│
├── smart_mail_agent/
│   └── ...
│
├── langgraph_concept.ipynb
│
├── requirements.txt
│
└── README.md
```

---

# 🎬 Project 1 — Movie Ticket Booking Agent

A complete **Agentic AI movie ticket booking application** built using Python, LangGraph/LangChain, LLMs and FastAPI.

The agent can understand natural-language requests and perform movie-related operations.

### ✨ Features

* 🎥 Show available movies
* 🔎 Get movie details
* 🎟️ Book movie tickets
* 👤 Collect customer information
* 🔢 Select number of seats
* 💰 Calculate booking amount
* 📋 Check booking status
* ❌ Cancel booking flow
* 🧠 Maintain conversation state
* 🤖 Use an LLM for intent understanding
* 🌐 FastAPI backend
* 💻 Web-based frontend

### Example

```text
User:
I am Pradeep, book 2 tickets for Avengers.

Agent:
I found Avengers.
There are available seats.
Would you like to confirm the booking?
```

The project demonstrates how an AI agent can combine **LLM reasoning + application logic + tools + state + APIs**.

---

# 📧 Project 2 — Smart Mail Agent

A smart AI agent designed to work with email-related tasks.

The project demonstrates how Agentic AI can be used to build applications that understand user requests and execute multi-step workflows.

### Concepts Demonstrated

* LLM-based understanding
* Agent workflows
* Tool calling
* State management
* Routing
* Email-related automation
* Structured responses

---

# 🕸️ LangGraph Concepts

The repository also contains practical examples for understanding **LangGraph**, including concepts such as:

```text
User Input
     ↓
     Router
     ↓
 ┌───┴────┐
 ↓        ↓
Agent    Tool
 ↓        ↓
 └───┬────┘
     ↓
   State
     ↓
 Final Response
```

LangGraph is particularly useful for building stateful and controllable agent workflows.

---

# 🧠 Agentic AI Architecture

A typical agent workflow in this repository can be represented as:

```text
                ┌──────────────┐
                │     User     │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │     LLM      │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │    Router    │
                └──────┬───────┘
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Tool          Agent       Database
          ↓            ↓            ↓
          └────────────┼────────────┘
                       ↓
                ┌──────────────┐
                │    State     │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │    Output    │
                └──────────────┘
```

---

# 🛠️ Tech Stack

| Technology          | Purpose                                      |
| ------------------- | -------------------------------------------- |
| Python              | Core programming                             |
| LangChain           | LLM application framework                    |
| LangGraph           | Stateful agent workflows                     |
| LLMs                | Reasoning and natural-language understanding |
| FastAPI             | Backend API                                  |
| SQLite              | Database / persistence                       |
| HTML/CSS/JavaScript | Web interface                                |
| Pydantic            | Data validation                              |
| Git & GitHub        | Version control                              |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/DataWithPdeep/agentic-ai-tutorial_With_project.git
```

```bash
cd agentic-ai-tutorial_With_project
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

If a project requires an LLM API key, create a `.env` file.

Example:

```env
OPENAI_API_KEY=your_api_key_here
```

⚠️ **Never commit your API keys or secrets to GitHub.**

Make sure `.env` is included in `.gitignore`.

---

# ▶️ Running the Movie Ticket Project

Navigate to the project:

```bash
cd movie_ticket
```

Start the FastAPI server:

```bash
uvicorn web_server:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 💡 Example User Queries

You can interact with the movie agent using natural language.

```text
Show all movies
```

```text
Details of Avengers
```

```text
I am Pradeep, book 2 tickets for Avengers
```

```text
Check the status of booking 1
```

The goal is to allow users to interact with the application naturally instead of manually navigating through multiple forms.

---

# 🎯 Learning Roadmap

This repository can be followed in the following order:

### 1️⃣ LLM Fundamentals

Understand:

* LLMs
* Prompts
* Structured outputs
* Model responses

### 2️⃣ LangChain

Learn:

* Models
* Prompt templates
* Chains
* Tools
* Structured output
* Memory

### 3️⃣ LangGraph

Learn:

* State
* Nodes
* Edges
* Conditional routing
* Workflows
* Checkpoints

### 4️⃣ Agentic AI

Build agents capable of:

* Reasoning
* Decision making
* Tool usage
* Multi-step execution
* State management

### 5️⃣ Real-World Projects

Apply the concepts to complete applications such as:

* 🎬 Movie Ticket Booking Agent
* 📧 Smart Mail Agent
* 🤖 AI-powered workflows

---

# 📌 Why Agentic AI?

Traditional software generally follows a predefined flow:

```text
Input → Code → Output
```

Agentic AI introduces a more dynamic workflow:

```text
Goal
 ↓
Understand
 ↓
Reason
 ↓
Choose Action
 ↓
Use Tool
 ↓
Observe Result
 ↓
Continue / Decide
 ↓
Final Response
```

This makes Agentic AI particularly useful for applications involving **multi-step tasks, tools, APIs, databases, and dynamic decision making**.

---

# 🎓 Who Is This Repository For?

This repository is useful for:

* Python developers
* Data Scientists
* ML Engineers
* AI Engineers
* GenAI developers
* Students learning Agentic AI
* Developers learning LangChain
* Developers learning LangGraph
* Anyone interested in building AI Agents

---

# 📚 Topics Covered

```text
Python
   ↓
LLMs
   ↓
LangChain
   ↓
Structured Output
   ↓
Tool Calling
   ↓
Memory
   ↓
LangGraph
   ↓
State Management
   ↓
Routing
   ↓
APIs
   ↓
Databases
   ↓
Agentic AI
   ↓
Real-World Projects
```

---

# 🚀 Future Improvements

Planned improvements may include:

* [ ] Multi-agent systems
* [ ] Advanced LangGraph workflows
* [ ] RAG integration
* [ ] Vector databases
* [ ] MCP integration
* [ ] Agent evaluation
* [ ] Observability
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Production-grade authentication
* [ ] Automated testing
* [ ] CI/CD pipeline

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you find a bug or have an idea for improving the projects:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Push the branch
6. Create a Pull Request

---

# ⭐ Support

If you find this repository useful for learning **Agentic AI, LangChain, or LangGraph**, consider giving the repository a ⭐.

---

## 👨‍💻 Author

**Pradeep Singh**

AI & Data Science | Generative AI | Agentic AI | Machine Learning | MLOps

GitHub: **DataWithPdeep**

---

## 📜 License

This project is intended for educational and learning purposes.

# -*- coding: utf-8 -*-
"""
Multi-Agent Architecture Practical Implementation
A multi-agent architecture consists of multiple agents working together to solve a problem or accomplish a task.

## Types of Multi-Agent Architectures:
* **Centralized**: One central controller coordinates the activities of the agents.
* **Decentralized**: No central controller; agents operate independently and communicate to solve tasks collaboratively.
* **Hybrid**: A combination of centralized and decentralized methods, where some coordination is central, and others are independent.

## Key Features
* **Autonomy**: Agents can act independently based on their programming or environment.
* **Cooperation**: Agents may collaborate or share information to achieve a common goal.
* **Adaptability**: Agents can learn from their environment or past actions and adjust their behavior accordingly.
"""

# Install required libraries (uncomment if needed)
# !pip install -q langchain langchain_groq langchain_community langgraph rizaio python-dotenv pinecone-client

import os
from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from pprint import pprint

# Load environment variables from config.env file
try:
    from dotenv import load_dotenv
    # Load environment variables from config.env file
    load_dotenv('config.env')
    print("Successfully loaded API keys from config.env")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    # Fallback: manually load the config file
    try:
        with open('config.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("Successfully loaded API keys from config.env (manual method)")
    except FileNotFoundError:
        print("config.env file not found. Please create it with your API keys.")
    except Exception as e:
        print(f"Error loading config.env: {e}")

# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
riza_api_key = os.getenv("RIZA_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Validate that all required API keys are present
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables or config.env")
if not riza_api_key:
    raise ValueError("RIZA_API_KEY not found in environment variables or config.env")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables or config.env")

print("All API keys loaded successfully!")

# Initialize ChatGroq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Test LLM connection
try:
    test_response = llm.invoke('Hi')
    print("LLM connection successful!")
except Exception as e:
    print(f"Error connecting to LLM: {e}")
    print("Please check your API keys and internet connection.")

# Define Tools
tool_tavily = TavilySearchResults(max_results=2)
tool_code_interpreter = ExecPython()
tools = [tool_tavily, tool_code_interpreter]

# Define Supervisor Agent
system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and RAG. Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focus on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information from external sources.
3. RAG: Specializes in retrieving information from the knowledge base and generating answers based on stored documents and context.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

**Routing Guidelines**:
- Use 'enhancer' for unclear or vague queries that need clarification
- Use 'researcher' for questions requiring current, real-time information from the web
- Use 'rag' for questions that can be answered using stored knowledge and documents

Your goal is to maximize accuracy and effectiveness by leveraging each agent's unique expertise while ensuring smooth workflow execution.
''')

class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "rag"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering from external sources, "
                    "'rag' for retrieving information from the knowledge base and generating contextual answers."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "rag"]]:
    """
    Supervisor node for routing tasks based on the current state and LLM response.
    Args:
        state (MessagesState): The current state containing message history.
    Returns:
        Command: A command indicating the next state or action.
    """
    try:
        # Prepare messages by appending the system prompt to the message history
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        # Invoke the language model with structured output using the Supervisor schema
        response = llm.with_structured_output(Supervisor).invoke(messages)
        goto = response.next
        reason = response.reason
        
        print(f"Current Node: Supervisor -> Goto: {goto}")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(content=reason, name="supervisor")
                ]
            },
            goto=goto,
        )
    except Exception as e:
        print(f"Error in supervisor_node: {e}")
        # Default to enhancer if there's an error
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Error in supervisor, defaulting to enhancer", name="supervisor")
                ]
            },
            goto="enhancer",
        )

def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer node for refining and clarifying user inputs.
    """
    try:
        system_prompt = (
            "You are an advanced query enhancer. Your task is to:\n"
            "Don't ask anything to the user, select the most appropriate prompt\n"
            "1. Clarify and refine user inputs.\n"
            "2. Identify any ambiguities in the query.\n"
            "3. Generate a more precise and actionable version of the original request.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        enhanced_query = llm.invoke(messages)
        print(f"Current Node: Prompt Enhancer -> Goto: Supervisor")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=enhanced_query.content,
                        name="enhancer"
                    )
                ]
            },
            goto="supervisor",
        )
    except Exception as e:
        print(f"Error in enhancer_node: {e}")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Error in enhancer", name="enhancer")
                ]
            },
            goto="supervisor",
        )

def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research node for leveraging a ReAct agent to process research-related tasks.
    """
    try:
        research_agent = create_react_agent(
            llm,
            tools=[tool_tavily],
            state_modifier="You are a researcher. Focus on gathering information and generating content. Do not perform any other tasks"
        )
        
        result = research_agent.invoke(state)
        print(f"Current Node: Researcher -> Goto: Validator")
        
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content,
                        name="researcher"
                    )
                ]
            },
            goto="validator",
        )
    except Exception as e:
        print(f"Error in research_node: {e}")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Error in research", name="researcher")
                ]
            },
            goto="validator",
        )

def rag_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    RAG (Retrieval-Augmented Generation) node for retrieving relevant context from Pinecone
    and generating answers based on the retrieved information with website-specific filtering.
    """
    try:
        # Import Pinecone utilities
        import sys
        sys.path.append('../Pinecone')
        from Pinecone_utils import initialize_pinecone, connect_to_index, query_vectors
        import pinecone
        
        # Initialize Pinecone connection
        if not initialize_pinecone():
            raise Exception("Failed to initialize Pinecone")
        
        # Connect to the index using Pinecone utilities (loads from pinecone.env)
        if not connect_to_index():
            raise Exception("Failed to connect to Pinecone index")
        
        # Extract the user's query and website context from the state
        user_query = None
        website_name = None
        
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'name') and msg.name == "user":
                user_query = msg.content
                # Check if query contains website specification
                if "website:" in user_query.lower():
                    parts = user_query.split("website:", 1)
                    if len(parts) > 1:
                        user_query = parts[0].strip()
                        website_name = parts[1].strip()
                break
        
        if not user_query:
            user_query = state["messages"][-1].content
        
        # Get the current index from Pinecone utils
        from Pinecone_utils import _client
        index = _client.current_index
        
        # Generate embedding for the query
        from langchain.embeddings import OpenAIEmbeddings
        embedder = OpenAIEmbeddings()
        query_embedding = embedder.embed_query(user_query)
        
        # Prepare filter for website-specific query
        filter_dict = None
        if website_name:
            filter_dict = {
                'website_name': {'$eq': website_name}
            }
            print(f"Filtering for website: {website_name}")
        
        # Query Pinecone with website-specific filter
        try:
            results = query_vectors(
                vector=query_embedding,
                top_k=5,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            matches = results.matches if hasattr(results, 'matches') else []
        except Exception as query_error:
            print(f"Error in Pinecone query: {query_error}")
            matches = []
        
        # Extract retrieved documents with website information
        retrieved_docs = []
        website_sources = set()
        
        for match in matches:
            if hasattr(match, 'metadata') and match.metadata:
                text = match.metadata.get('text', '')
                website = match.metadata.get('website_name', 'Unknown')
                source_url = match.metadata.get('source_url', '')
                
                retrieved_docs.append(text)
                website_sources.add(website)
                
                print(f"Retrieved from {website}: {source_url}")
        
        # If no documents found, provide a fallback response
        if not retrieved_docs:
            if website_name:
                response_content = f"I couldn't find relevant information for '{user_query}' in the knowledge base for website '{website_name}'. Please try rephrasing your question or ask about a different topic."
            else:
                response_content = f"I couldn't find relevant information in the knowledge base for: '{user_query}'. Please try rephrasing your question or ask about a different topic."
        else:
            # Build context from retrieved documents
            context = "\n\n".join(retrieved_docs)
            
            # Create a prompt for the LLM
            if website_name:
                rag_prompt = f"""Based on the following retrieved information from website '{website_name}', please answer the user's question. 
                If the information is not sufficient to answer the question completely, acknowledge what you can answer and what additional information might be needed.

                Retrieved Information from {website_name}:
                {context}

                User Question: {user_query}

                Answer:"""
            else:
                rag_prompt = f"""Based on the following retrieved information, please answer the user's question. 
                If the information is not sufficient to answer the question completely, acknowledge what you can answer and what additional information might be needed.

                Retrieved Information:
                {context}

                User Question: {user_query}

                Answer:"""
            
            # Generate response using the LLM
            llm_response = llm.invoke(rag_prompt)
            response_content = llm_response.content
        
        print(f"Current Node: RAG -> Goto: validator")
        print(f"Retrieved {len(retrieved_docs)} documents from {len(website_sources)} website(s)")

        return Command(
            update={
                "messages": [
                    HumanMessage(content=response_content, name="rag")
                ]
            },
            goto="validator",
        )
    except Exception as e:
        print(f"Error in rag_node: {e}")
        return Command(
            update={
                "messages": [
                    HumanMessage(content=f"Error in RAG processing: {str(e)}", name="rag")
                ]
            },
            goto="validator",
        )

# Validator system prompt
validator_system_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

Routing Guidelines:
1. 'supervisor' Agent: For unclear or vague state messages.
2. Respond with 'FINISH' to end the workflow.
'''

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator node for checking if the question and the answer are appropriate.
    """
    try:
        # Extract the first (user's question) and the last (agent's response) messages
        user_question = state["messages"][0].content
        agent_answer = state["messages"][-1].content

        messages = [
            {"role": "system", "content": validator_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": agent_answer},
        ]

        response = llm.with_structured_output(Validator).invoke(messages)
        goto = response.next
        reason = response.reason

        if goto == "FINISH":
            goto = END
            print("Transitioning to END")
        else:
            print(f"Current Node: Validator -> Goto: Supervisor")

        return Command(
            update={
                "messages": [
                    HumanMessage(content=reason, name="validator")
                ]
            },
            goto=goto,
        )
    except Exception as e:
        print(f"Error in validator_node: {e}")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Error in validation", name="validator")
                ]
            },
            goto=END,
        )

# Build the workflow graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("enhancer", enhancer_node)
builder.add_node("researcher", research_node)
builder.add_node("rag", rag_node)
builder.add_node("validator", validator_node)

# Add edges
builder.add_edge(START, "supervisor")

# Compile the graph
graph = builder.compile()

def run_workflow(user_input: str):
    """
    Run the multi-agent workflow with a user input.
    
    Args:
        user_input (str): The user's question or request
        
    Returns:
        List of outputs from each node in the workflow
    """
    inputs = {
        "messages": [
            HumanMessage(content=user_input, name="user"),
        ]
    }
    
    outputs = []
    for output in graph.stream(inputs):
        for key, value in output.items():
            if value is None:
                continue
            outputs.append((key, value))
            print(f"Output from node '{key}':")
            pprint(value, indent=2, width=80, depth=None)
            print()
    
    return outputs

# Example usage
if __name__ == "__main__":
    # Test the workflow with different types of queries
    test_queries = [
        "What's the weather in Hyderabad today?",
        "What is the difference between the stock price of Apple in 2023 and 2021?",
        "Research the impact of climate change on agriculture in Southeast Asia",
        "How many A's are present in the string 'AVYGABAAHKJHDAAAAUHBU'?",
        "Tell me about machine learning algorithms",  # This would use RAG if you have ML docs in Pinecone
        "What are the best practices for data preprocessing?",  # This would use RAG if you have data science docs in Pinecone
        "What is machine learning? website: Example Website",  # Website-specific RAG query
        "How does the company handle customer support? website: Company Website"  # Another website-specific query
    ]
    
    print("Multi-Agent Architecture Workflow")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)
        try:
            run_workflow(query)
        except Exception as e:
            print(f"Error running workflow: {e}")
        print("\n" + "=" * 50)




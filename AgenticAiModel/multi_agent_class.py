# -*- coding: utf-8 -*-
"""
Multi-Agent Architecture Class Implementation
A class-based implementation of the multi-agent workflow that accepts website name and query.
"""

import os
from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from pprint import pprint

class MultiAgentWorkflow:
    """
    Multi-Agent Workflow Class that manages a team of agents to process user queries.
    """
    
    def __init__(self, website_name: str = None, debug_mode: bool = False):
        """
        Initialize the Multi-Agent Workflow.
        
        Args:
            website_name (str): The website name to use for RAG queries (optional)
            debug_mode (bool): If True, shows detailed workflow information. If False, shows only final answer.
        """
        self.website_name = website_name
        self.debug_mode = debug_mode
        self.llm = None
        self.tools = None
        self.graph = None
        
        # Initialize node usage counters
        self.node_counters = {
            "researcher": 0,
            "rag": 0,
            "enhancer": 0,
            "web_search": 0
        }
        
        # Load environment variables
        self._load_environment()
        
        # Initialize components
        self._initialize_llm()
        self._initialize_tools()
        self._build_workflow()
    
    def _load_environment(self):
        """Load environment variables from config files."""
        try:
            from dotenv import load_dotenv
            # Load environment variables from config files
            load_dotenv('../Pinecone/pinecone.env')
            load_dotenv('config.env')
            print("Successfully loaded API keys from config files")
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
        
        # Validate API keys
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate that all required API keys are present."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        riza_api_key = os.getenv("RIZA_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or config.env")
        if not riza_api_key:
            raise ValueError("RIZA_API_KEY not found in environment variables or config.env")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables or config.env")
        
        print("All API keys loaded successfully!")
    
    def _initialize_llm(self):
        """Initialize the ChatGroq LLM."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
        # Test LLM connection
        try:
            test_response = self.llm.invoke('Hi')
            print("LLM connection successful!")
        except Exception as e:
            print(f"Error connecting to LLM: {e}")
            print("Please check your API keys and internet connection.")
    
    def _initialize_tools(self):
        """Initialize the tools for research agent."""
        tool_tavily = TavilySearchResults(max_results=2)
        tool_web = DuckDuckGoSearchRun()
        self.tools = [tool_tavily, tool_web]
    
    def _build_workflow(self):
        """Build the workflow graph with all nodes."""
        # Define the workflow graph
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("enhancer", self._enhancer_node)
        builder.add_node("researcher", self._research_node)
        builder.add_node("web_search", self._web_search_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("validator", self._validator_node)
        
        # Add edges
        builder.add_edge(START, "supervisor")
        
        # Compile the graph
        self.graph = builder.compile()
    
    def _supervisor_node(self, state: MessagesState) -> Command[Literal["enhancer", "researcher", "web_search", "rag"]]:
        """Supervisor node for routing tasks to appropriate agents."""
        try:
            # Extract the user's question from the state
            user_question = state["messages"][-1].content
            
            # Define supervisor system prompt
            system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and RAG. Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focus on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information from external sources using Tavily search.
3. Web Search: Specializes in DuckDuckGo web search for real-time information and current events.
4. RAG: Specializes in retrieving information from the knowledge base and generating answers based on stored documents and context.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

**Routing Guidelines**:
- Use 'enhancer' for unclear or vague queries that need clarification
- Use 'researcher' for questions requiring current, real-time information from the web (news, weather, live data) using Tavily search
- Use 'web_search' for DuckDuckGo web search queries, breaking news, and real-time information
- Use 'rag' for questions about stored knowledge, courses, features, or information that can be found in the knowledge base
- Questions about courses, features, or general information about a specific website should go to RAG
''')

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ]

            response = self.llm.with_structured_output(Supervisor).invoke(messages)
            next_agent = response.next
            reason = response.reason

            # Check if research node has been used more than 2 times
            if next_agent == "researcher" and self.node_counters["researcher"] >= 2:
                if self.debug_mode:
                    print(f"Research node used {self.node_counters['researcher']} times, redirecting to RAG")
                next_agent = "rag"
                reason = f"Research node has been used {self.node_counters['researcher']} times. Redirecting to RAG for alternative information retrieval."
            
            # Check if web search node has been used more than 2 times
            if next_agent == "web_search" and self.node_counters["web_search"] >= 2:
                if self.debug_mode:
                    print(f"Web search node used {self.node_counters['web_search']} times, redirecting to researcher")
                next_agent = "researcher"
                reason = f"Web search node has been used {self.node_counters['web_search']} times. Redirecting to researcher for alternative information gathering."
            
            # Check if RAG node has been used more than 2 times
            if next_agent == "rag" and self.node_counters["rag"] >= 2:
                if self.debug_mode:
                    print(f"RAG node used {self.node_counters['rag']} times, redirecting to web search")
                next_agent = "web_search"
                reason = f"RAG node has been used {self.node_counters['rag']} times. Redirecting to web search for real-time information."

            if self.debug_mode:
                print(f"Current Node: Supervisor -> Goto: {next_agent}")
                print(f"Reason: {reason}")
                print(f"Node usage counters: {self.node_counters}")

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=reason, name="supervisor")
                    ]
                },
                goto=next_agent,
            )
        except Exception as e:
            print(f"Error in supervisor_node: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content="Error in supervision", name="supervisor")
                    ]
                },
                goto="enhancer",
            )
    
    def _enhancer_node(self, state: MessagesState) -> Command[Literal["supervisor"]]:
        """Enhancer node for improving user queries."""
        try:
            # Increment enhancer counter
            self.node_counters["enhancer"] += 1
            
            user_question = state["messages"][0].content

#             enhancer_system_prompt = '''
# You are a prompt enhancer. Your task is to improve user queries by:
# 1. Clarifying vague or ambiguous language
# 2. Adding context where needed
# 3. Breaking down complex questions into simpler parts
# 4. Ensuring the query is specific and actionable

# Provide an enhanced version of the user's question that is clearer and more specific.
# '''
            system_prompt = (
                "You are an advanced query enhancer. Your task is to:\n"
                "Don't ask anything to the user, select the most appropriate prompt"
                "1. Clarify and refine user inputs.\n"
                "2. Identify any ambiguities in the query.\n"
                "3. Generate a more precise and actionable version of the original request.\n"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ]

            enhanced_question = self.llm.invoke(messages).content

            if self.debug_mode:
                print(f"Current Node: Enhancer -> Goto: supervisor")
                print(f"Enhanced question: {enhanced_question}")
                print(f"Enhancer used {self.node_counters['enhancer']} times")

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=enhanced_question, name="enhancer")
                    ]
                },
                goto="supervisor",
            )
        except Exception as e:
            print(f"Error in enhancer_node: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content="Error in enhancement", name="enhancer")
                    ]
                },
                goto="supervisor",
            )
    
    def _research_node(self, state: MessagesState) -> Command[Literal["validator"]]:
        """
        Research node for leveraging a ReAct agent to gather information using web search.

        Args:
            state (MessagesState): The current state containing the conversation history.

        Returns:
            Command: A command to update the state with the research results and route to the validator.
        """
        try:
            # Increment researcher counter
            self.node_counters["researcher"] += 1
            
            # Create a specialized ReAct agent for research tasks
            research_agent = create_react_agent(
                self.llm,
                tools=[self.tools[0]],  # Use the Tavily tool
                state_modifier=(
                    "You are a researcher. Focus on gathering accurate and relevant information "
                    "using web search. Provide comprehensive answers based on your findings."
                )
            )

            # Invoke the agent with the current state to process the input and perform research
            result = research_agent.invoke(state)

            # Debug logging to trace responses and node transitions
            if self.debug_mode:
                print(f"Current Node: Researcher -> Goto: validator")
                print(f"Researcher used {self.node_counters['researcher']} times")
                # print(f"Response:", result)

            # Handle different result formats
            if isinstance(result, dict) and "messages" in result and result["messages"]:
                content = result["messages"][-1].content
            elif isinstance(result, dict) and "output" in result:
                content = result["output"]
            elif hasattr(result, "output"):
                content = result.output
            elif hasattr(result, "messages") and result.messages:
                content = result.messages[-1].content
            else:
                content = str(result)

            # Return a command to update the state and move to the 'validator' node
            return Command(
                update={
                    "messages": [
                        # Append the last message (agent's response) to the state, tagged with "researcher"
                        HumanMessage(content=content, name="researcher")
                    ]
                },
                # Specify the next node in the workflow: "validator"
                goto="validator",
            )
        except Exception as e:
            print(f"Error in research_node: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"Error in research: {str(e)}", name="researcher")
                    ]
                },
                goto="validator",
            )
    
    def _web_search_node(self, state: MessagesState) -> Command[Literal["validator"]]:
        """
        Web Search node for direct web search using DuckDuckGo search.

        Args:
            state (MessagesState): The current state containing the conversation history.

        Returns:
            Command: A command to update the state with the web search results and route to the validator.
        """
        try:
            # Increment web search counter
            self.node_counters["web_search"] += 1
            
            # Extract the user's query from the state
            user_query = state["messages"][0].content
            
            # Create a specialized ReAct agent for web search tasks
            web_search_agent = create_react_agent(
                 self.llm,
                 tools=[self.tools[1]],  # Use the DuckDuckGo search tool
                 state_modifier=(
                     "You are a web search specialist. Focus on finding real-time information, "
                     "breaking news, and current events using DuckDuckGo search. "
                     "Provide comprehensive answers based on the latest web search results."
                 )
             )

            # Invoke the agent with the current state to process the input and perform web search
            result = web_search_agent.invoke(state)

            # Debug logging to trace responses and node transitions
            if self.debug_mode:
                print(f"Current Node: Web Search -> Goto: validator")
                print(f"Web Search used {self.node_counters['web_search']} times")
                # print(f"Response:", result)

            # Handle different result formats
            if isinstance(result, dict) and "messages" in result and result["messages"]:
                content = result["messages"][-1].content
            elif isinstance(result, dict) and "output" in result:
                content = result["output"]
            elif hasattr(result, "output"):
                content = result.output
            elif hasattr(result, "messages") and result.messages:
                content = result.messages[-1].content
            else:
                content = str(result)

            # Return a command to update the state and move to the 'validator' node
            return Command(
                update={
                    "messages": [
                        # Append the last message (agent's response) to the state, tagged with "web_search"
                        HumanMessage(content=content, name="web_search")
                    ]
                },
                # Specify the next node in the workflow: "validator"
                goto="validator",
            )
        except Exception as e:
            print(f"Error in web_search_node: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"Error in web search: {str(e)}", name="web_search")
                    ]
                },
                goto="validator",
            )
    
    def _rag_node(self, state: MessagesState) -> Command[Literal["validator"]]:
        """RAG node for retrieving relevant context from Pinecone."""
        try:
            # Increment RAG counter
            self.node_counters["rag"] += 1
            
            # Import Pinecone utilities
            import sys
            sys.path.append('../Pinecone')
            from Pinecone_utils import initialize_pinecone, connect_to_index, query_vectors, generate_embedding
            
            # Initialize Pinecone connection
            if not initialize_pinecone():
                raise Exception("Failed to initialize Pinecone")
            
            # Connect to the index using Pinecone utilities (loads from pinecone.env)
            if not connect_to_index():
                raise Exception("Failed to connect to Pinecone index")
            
            # Extract the user's query from the state
            user_query = state["messages"][0].content
            if self.debug_mode:
                print(f"User Query: {user_query}")
            
            # Use the website name from the class instance
            website_name = self.website_name
            if not website_name:
                raise Exception("Website name not provided. Please set website_name when initializing the class.")
            
            if self.debug_mode:
                print(f"Searching in website: {website_name}")
            
            # Generate embedding for the user query
            query_embedding = generate_embedding(user_query)
            if self.debug_mode:
                print(f"Generated embedding for query: {len(query_embedding)} dimensions")
            
            # Prepare filter for website-specific query
            filter_dict = {
                'website_name': {'$eq': website_name}
            }
            
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
                if self.debug_mode:
                    print(f"Found {len(matches)} relevant chunks")
            except Exception as query_error:
                if self.debug_mode:
                    print(f"Error in Pinecone query: {query_error}")
                matches = []
            
            # Extract retrieved documents
            retrieved_docs = []
            
            for i, match in enumerate(matches):
                if hasattr(match, 'metadata') and match.metadata:
                    text = match.metadata.get('text', '')
                    retrieved_docs.append(text)
                    if self.debug_mode:
                        print(f"Chunk {i+1} retrieved (Score: {match.score:.4f})")

            # If no documents found, provide a fallback response
            if not retrieved_docs:
                response_content = f"I don't have specific information about '{user_query}' in the {website_name} knowledge base. You might want to try rephrasing your question or ask about a different topic related to {website_name}."
            else:
                # Build context from retrieved documents
                context = "\n\n".join(retrieved_docs)
                
                # Create a comprehensive prompt for the LLM
                rag_prompt = f"""You are a helpful AI assistant that provides direct, natural answers to user questions. You have access to information from the website '{website_name}' to help answer questions.

RELEVANT INFORMATION FROM {website_name}:
{context}

USER QUESTION: {user_query}

INSTRUCTIONS:
1. Answer the user's question directly and naturally, as if you're having a conversation
2. Use the provided information to give a comprehensive and accurate response
3. Don't mention "based on the context" or "according to the information" - just answer naturally
4. If the information is sufficient, provide a complete answer
5. If information is missing, acknowledge what you can answer and suggest what else might be helpful
6. Be conversational and helpful in your tone
7. Structure your response clearly and logically
8. If you can't answer from the available information, say so directly

ANSWER:"""
                
                # Generate response using the LLM
                llm_response = self.llm.invoke(rag_prompt)
                response_content = llm_response.content
            
            if self.debug_mode:
                print(f"Current Node: RAG -> Goto: validator")
                print(f"Retrieved {len(retrieved_docs)} documents from {website_name}")
                print(f"RAG used {self.node_counters['rag']} times")

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
    
    def _validator_node(self, state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
        """
        Validator node for checking if the question and the answer are appropriate.

        Args:
            state (MessagesState): The current state containing message history.

        Returns:
            Command: A command indicating whether to route back to the supervisor or end the workflow.
        """
        try:
            # Extract the first (user's question) and the last (agent's response) messages
            user_question = state["messages"][0].content
            agent_answer = state["messages"][-1].content

            # System prompt providing clear instructions to the validator agent
            system_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.

CRITICAL VALIDATION CRITERIA - REJECT THE ANSWER IF:
1. The answer says "I couldn't find information" or "I don't have information"
2. The answer asks the user for more details or clarification
3. The answer mentions the wrong person/topic (e.g., talking about Piyush when asked about Mahesh Babu)
4. The answer is vague, incomplete, or doesn't directly address the question
5. The answer suggests trying a different approach or rephrasing the question
6. The answer contains phrases like "I'd be happy to help you find it" or "I'd need more context"

ACCEPT THE ANSWER IF:
1. It directly and comprehensively answers the user's question
2. It provides specific, relevant information
3. It addresses the exact topic/person mentioned in the question
4. It's complete and doesn't require additional input from the user

Routing Guidelines:
- Route to 'supervisor' if the answer is unsatisfactory according to the criteria above
- Respond with 'FINISH' only if the answer satisfactorily addresses the question
'''

            # Prepare the message history with the system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": agent_answer},
            ]

            # Invoke the LLM with structured output using the Validator schema
            response = self.llm.with_structured_output(Validator).invoke(messages)

            # Extract the 'next' routing decision and the 'reason' from the response
            goto = response.next
            reason = response.reason

            # Debug logging to trace responses and transitions
            if self.debug_mode:
                print(f"Validator decision: {goto}")
                print(f"Validator reason: {reason}")
                # print(f"Response: {response}")

            # Determine the next node in the workflow
            if goto == "FINISH" or goto == END:
                goto = END  # Transition to the termination state
                if self.debug_mode:
                    print("Transitioning to END")  # Debug log to indicate process completion
            else:
                if self.debug_mode:
                    print(f"Current Node: Validator -> Goto: Supervisor")  # Log for routing back to supervisor

            # Return a command with the updated state and the determined routing destination
            return Command(
                update={
                    "messages": [
                        # Append the reason (validator's response) to the state, tagged with "validator"
                        HumanMessage(content=reason, name="validator")
                    ]
                },
                goto=goto,  # Specify the next node in the workflow
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
    
    def _reset_counters(self):
        """Reset node usage counters for a new query."""
        self.node_counters = {
            "researcher": 0,
            "rag": 0,
            "enhancer": 0,
            "web_search": 0
        }
        if self.debug_mode:
            print("Node counters reset for new query")
    
    def run_query(self, query: str) -> List:
        """
        Run a query through the multi-agent workflow.
        
        Args:
            query (str): The user's question or request
            
        Returns:
            List: List of outputs from each node in the workflow
        """
        # Reset counters for new query
        self._reset_counters()
        
        inputs = {
            "messages": [
                HumanMessage(content=query, name="user"),
            ]
        }
        
        outputs = []
        final_answer = None
        
        if self.debug_mode:
            print("Starting workflow execution...")
            print(f"Initial node counters: {self.node_counters}")
        
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                if value is None:
                    continue
                outputs.append((key, value))
                
                # Store the final answer from the last agent (not validator)
                if key in ["researcher", "rag", "web_search"] and value and "messages" in value:
                    final_answer = value["messages"][-1].content
                
                if self.debug_mode:
                    print(f"Output from node '{key}':")
                    pprint(value, indent=2, width=80, depth=None)
                    print()
                    print(f"Node '{key}' completed. Moving to next node...")
                    print(f"Current node counters: {self.node_counters}")
                    print("-" * 40)
        
        # Show final answer in normal mode
        if not self.debug_mode and final_answer:
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(final_answer)
            print("="*60)
        
        return outputs

# Pydantic models for structured output
class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "web_search", "rag"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering from external sources using Tavily search, "
                    "'web_search' for DuckDuckGo web search for real-time information and current events, "
                    "'rag' for retrieving information from the knowledge base and generating contextual answers."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

# Example usage
if __name__ == "__main__":
    print("Multi-Agent Architecture Workflow (Class Version)")
    print("=" * 60)
    
    # Initialize the workflow with website name
    website_name = input("Enter website name (e.g., Piyush, Elancode): ").strip()
    if not website_name:
        website_name = "Piyush"  # Default
    
    # Ask for debug mode
    debug_input = input("Enable debug mode? (y/n): ").strip().lower()
    debug_mode = debug_input in ['y', 'yes', '1', 'true']
    
    workflow = MultiAgentWorkflow(website_name=website_name, debug_mode=debug_mode)
    
    if debug_mode:
        print(f"\nWorkflow initialized for website: {website_name} (DEBUG MODE)")
        print("Enter 'quit' to exit")
        print("-" * 60)
    else:
        print(f"\nWorkflow initialized for website: {website_name}")
        print("Enter 'quit' to exit")
        print("-" * 60)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        if debug_mode:
            print(f"\nProcessing query: {query}")
            print("-" * 40)
        
        try:
            outputs = workflow.run_query(query)
            if debug_mode:
                print(f"\nQuery completed. Total outputs: {len(outputs)}")
        except Exception as e:
            print(f"Error running workflow: {e}")
        
        if debug_mode:
            
            print("=" * 60) 
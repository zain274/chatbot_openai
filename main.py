import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


@cl.on_chat_start
async def start():
    #Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    """Set up the chat session when a user connects."""
    # Initialize an empty chat history in the session.
    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)
    agent: Agent = Agent(name="Assistant", instructions="FINAL SYSTEM PROMPT — TECH WAGERA (Ultimate Version — Lowest Price Only) You are Tech Wagera, an AI product expert chatbot. You fetch product information from internal databases and calculate the lowest price using 5 external store sources internally. 🔒 Never reveal or mention store names or any internal/external data source. 1. Conversation Start Always begin with: “Hi, I am Tech Wagera Agent! Can I have your name, email, and contact number so I can assist you better?” You MUST NOT continue until user provides: Name Email Contact Number 2. After Getting User Details → Show Categories Show: 💻 Laptop 📱 Mobile 🎧 Headphones 🖨️ Other Items 3. Product Search Rules When user selects a category: ✔ Search internal product data ✔ Clean & normalize text ✔ Remove duplicates ✔ Match closest correct product ✔ Never guess If no match → apply Limited Stock Rule. Query Cleaning Examples “macboook aaiirr 5000” → MacBook Air “hp i7 12gen” → HP Core i7 12th Gen “ryzn 5 5600” → Ryzen 5 5600 “xiome note forteen” → Xiaomi Note 14 4. Beginner vs Normal User Detection (Corrected) ✔ If user provides NO specs, such as: “Laptop batao”, “Kaisa laptop?”, “Mujhe laptop chahiye”, → Ask use-type category questions. ✔ If user provides ANY specs, even basic: “i5 11th gen”, “Ryzen 5”, “8GB RAM”, “512 SSD”, “gaming laptop i5 11 gen 8GB 512SSD” → Do NOT ask category-type questions. → Go directly to product identification. 5. Category Type Rule (Corrected) Ask the following questions ONLY when the user gives zero specs: 💻 Laptop “What type of laptop do you want? 🕹️ Gaming 👨‍💻 Coding / Editing 📝 Normal Use” Collect: Processor, RAM, Storage, GPU. 📱 Mobile Ask type (Gaming / Normal / Camera Focus) ONLY if no specs given. Collect: RAM, Storage, Screen size (optional). 🎧 Headphones Ask: Gaming / Normal / Wireless (only if no specs given). 🖨️ Other Items Ask: Printer / Keyboard / Mouse / Monitor / Accessories (only if no specs given). If specs already provided → skip questions → identify product directly. 6. Response Rules (Lowest Price Only) ✔ For Laptops — show ONLY: Model Name CPU RAM Storage GPU Display Price: Rs XX,XXX (Lowest Price Only) ✔ For Mobiles / Headphones / Other Items: Model Name Short Specs Price: Rs XX,XXX (Lowest Price Only) ❌ NO price range ❌ NO store names ❌ NO store links ❌ NO data source mentions ❌ NO exact store prices 7. Store Source Visibility Rule Never reveal: Store names Store prices Price differences Which store is cheapest Any data source Only show: “Price: Rs XX,XXX” 8. More Information Rule If user asks for: Exact price Store-wise price Comparison Links Images Full specs Availability Reply ONLY: “For more information, please contact us on WhatsApp: https://wa.me/923213240204” 9. Limited Stock Rule If product cannot be found: “This product seems short. Please contact us on WhatsApp: https://wa.me/923213240204” 10. Off-Topic Rule If user asks anything non-product related: “Tech Wagera only provides product details and prices.” 11. Formatting Rules ✔ Use bullet points ✔ Keep replies short ✔ Show ONLY the lowest price ❌ No store names ❌ No external links (except WhatsApp) 12. Sample Output Redragon H320 Lamia 2 7.1 Surround RGB Lighting Noise-Cancel Mic USB Interface Price: Rs XX,XXX 13. Lowest Price Rule (Strict) When user asks for lowest price: ✔ Show ONLY: “Price: Rs XX,XXX” ❌ No price range ❌ No store names ❌ No store comparison If user demands store-based lowest price: “For more information, please contact us on WhatsApp: https://wa.me/923213240204”", model=model)
    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to the Tech wagera AI Assistant! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    # Create a new message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, history, run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Append the assistant's response to the history.
        history.append({"role": "assistant", "content": msg.content})

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")
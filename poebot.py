from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PoeBOT.ai", page_icon="/content/icon.jpeg")
st.title('Poebot-AI| AI Content Writer')

st.header('Cold Email Generator')
tab1 = st.tabs(["📈 Talk Here"])

uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing the document..."):
        temp_dir = "/tmp"  # Temporary directory

        # Ensure the temporary directory exists
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, "uploaded_file.pdf")

        # Check if a file is uploaded and temporarily store it
        if uploaded_file:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

        if os.path.exists(temp_path):
            loader = PyPDFLoader(temp_path)
            pages = loader.load_and_split()

            text_splitter = CharacterTextSplitter(
                chunk_size=800, chunk_overlap=50)
            documents = text_splitter.split_documents(pages)

            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')

            db = FAISS.from_documents(documents, embeddings)

            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "include_metadata": True}
            )

            prompt_template = """
            Your task is to generate effective and captivating cold emails that give higher response rates. \
            research about the company the user wants to contact. Make sure to state why the user would be a great fit for the position, based on your research.\
            Also state why the user wants to be considered for the job and how he can contribute to the company. State why the recipient should care about the email. \
            Your response should be concise, easy, and actionable and within a 100-word limit.
            Context: \n {context}?\n
            question: \n {question} \n

            Answer:
            """

            messages = [
                SystemMessagePromptTemplate.from_template(prompt_template),
            ]

            qa_prompt = ChatPromptTemplate.from_messages(messages)

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=False,
                combine_docs_chain_kwargs={'prompt': qa_prompt},
                verbose=False
            )

            st.success("Document processing completed!")

            # Container for the user's text input
            container = st.container()

            def generate_response(query):
                # Provide an empty list for chat_history
                result = chain(inputs={"question": query, "chat_history": []})
                return result["answer"]

            with container:
                with st.form(key="my_form", clear_on_submit=True):
                    user_input = st.text_input("You:", key="input")
                    submit_button = st.form_submit_button(label="Send")

                    if user_input and submit_button:
                        output = generate_response(user_input)
                        message(user_input, is_user=True, key="user_input",
                                avatar_style="adventurer")
                        message(output, key="output")

            # Response container for displaying messages
            response_container = st.container()
        else:
            st.error("Uploaded file does not exist at the expected location.")


st.header("Twitter/LinkedIn Post Generator")
tab2 = st.tabs(["📈 Talk Here"])

llm = OpenAI(temperature=0.5)

# Input section
input_section = st.empty()
with input_section:
    st.subheader("Input")
    post_description = st.text_input("Describe your post:")

# Output section
output_section = st.empty()
with output_section:
    st.subheader("Output")

    if st.button("Generate Post"):
        if post_description:
            prompt = PromptTemplate(
                input_variables=['post_description'],
                template='''As a seasoned social media strategist, your mission is to create a compelling Twitter/LinkedIn post based \
                on the given description: "{post_description}". Craft an engaging message with atleast a 200-word limit that resonates \
                with a broad audience aged 18-35. Evoke emotions, spark curiosity, and encourage interactions. Please generate 3 unique \
                and relevant hashtags to enhance discoverability. Generate the hashtags at the end of the post. Remember to be creative, \
                witty, and relatable while delivering a clear and captivating message.'''
            )

            chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
            generated_post = chain.run(post_description)

            # Display the generated post
            st.write(generated_post)
        else:
            st.warning("Please enter a post description.")


st.header("College Essay/Letter generator ")
tab3 = st.tabs(["📈 Talk Here"])

oneshotessay_template = """Your task is to generate strong and persuasive college essays based on the information provided by the user.
                            Use a framework that is similar to essays of students who have been accepted into ivy league colleges or MIT, Stanford.
                            There should be an 80 word limit to the essay. It must be concise and should also use intellegent language that\
                            captivate the reader. Try not to repeat the same words. Also include a conclusion that reaffirms the student's vehemence to attend the particular college.



                            Here is a question:
                            {input}"""

oneshotletter_template = """Your task is to generate a letter to the head of admissions/admissions office based on the information provided by the user. Use\
                            a framework that is similar to students who were waitlisted from top schools and were accepted after posting the letter.\
                            The word limit is 80 words. The letter must be short,simple and to the point. Try not to repeat the same words too many times.\
                            Showcase the users desire to attend the provided college by making the letter expressive.


                            Here is a question:
                            {input}"""

prompt_infos = [
    {
        "name": "oneshotEssay",
        "description": "The perfect tool to create an Ivy standard Essay",
        "prompt_template": oneshotessay_template
    },

    {
        "name": "oneshotLetter",
        "description": "The perfect tool to create an Ivy standard Letter to the Admissions",
        "prompt_template": oneshotletter_template
    },
]

llm = OpenAI()

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")


destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
)

user_input = st.text_area("Enter your text here:")

if st.button("Generate"):
    if user_input:
        # No need to specify a destination_name
        generated_text = chain.run(input=user_input, destination_name=None)
        # return generated_text
        st.subheader("Generated Content:")
        st.write(generated_text)
    else:
        st.warning("Please enter some text.")


st.header("Blog Post Generator")
tab4 = st.tabs(["📈 Talk Here"])

llm = OpenAI(temperature=0.8)

# Input section
input_section = st.empty()
with input_section:
    st.subheader("Input")
    blog_description = st.text_input("Describe your blog post:")

# Output section
output_section = st.empty()
with output_section:
    st.subheader("Output")

    if st.button("Generate Blog"):
        if blog_description:
            prompt = PromptTemplate(
                input_variables=['blog_description'],
                template='''Your task is to generate a compelling and informative blog post on the topic of {blog_description}. Also give an appropriate \
                title for the blog.Ensure the content is engaging, well-researched, and provides valuable insights or information to the reader. \
                Keep it concise and it should be around 180-190 words.'''
            )

            chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
            generated_post = chain.run(blog_description)

            # Display the generated post
            st.write(generated_post)
        else:
            st.warning("Please enter a post description.")

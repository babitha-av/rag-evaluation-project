from app.rag_pipeline import RAGPipeline


rag = RAGPipeline()


questions = [
    "How long is the refund period?",
    "How much does the annual plan cost?",
    "Does AcmeAI support multi-factor authentication?",
    "Can I cancel my subscription?",
    "Does cancellation automatically give me a refund?",
    "Who is the CEO of AcmeAI?"
]


for question in questions:

    result = rag.ask(question)

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(source)

    print("\nCONTEXT:")
    for context in result["context"]:
        print(context)

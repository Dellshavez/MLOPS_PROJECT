## Q&A Résumé PDF – 100% Open Source
Ce projet est une application web interactive développée avec Streamlit permettant de poser des questions en langage naturel sur le contenu d’un fichier PDF,
comme un cours ou un document technique. Grâce à l’utilisation de modèles de LLM open source, d’embeddings performants et de la recherche vectorielle,
il est possible d'obtenir des réponses précises à partir du contenu extrait

# Fonctionnalités:
Upload de fichiers PDF
Résumé et compréhension automatique des documents
Extraction de réponses via un LLM local (Flan-T5)
Recherche contextuelle à l’aide de FAISS + embeddings HuggingFace
100% exécutable en local (aucun appel à une API externe)



# Technologies utilisées:
Python
Streamlit
LangChain
HuggingFace Transformers & Embeddings
FAISS
PyPDF2
Flan-T5 (modèle allégé pour CPU)


# voici les paquets à inclure
streamlit
PyPDF2
transformers
langchain
faiss-cpu
sentence-transformers


# lancer l'application 
streamlit run rag_streamlit1.py





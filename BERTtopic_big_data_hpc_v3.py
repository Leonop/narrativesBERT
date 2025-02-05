if __name__ == "__main__":
    bt = BERTopicGPU()
    docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{gl.START_YEAR}_{gl.YEAR_FILTER}.txt')
    
    if os.path.exists(docs_path):
        print("Reading preprocessed docs from preprocessed_docs.txt")
        docs = bt.load_doc_parallel(docs_path)
        docs = list(set(docs))
    else:
        meta = bt.load_data()
        docs = bt.pre_process_text(meta)
        bt.save_file(docs, docs_path, bar_length=100)

    # Skip optimization and use predefined parameters
    topic_model = bt.Bertopic_run(docs)
    bt.save_topic_keywords(topic_model)
    bt.save_figures(topic_model)
    print("BERTopic model training completed.")
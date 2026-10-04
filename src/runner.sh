#!/usr/bin/env bash

echo "Please select an option:"
echo "1. rag_generation.py"
echo "2. evaluate_rag.py"
echo "3. build_vectorstore.py"
echo "4. testset_generator.py"

read -p "Enter your choice: " choice

case "$choice" in
    1)
        PYTHONPATH=. caffeinate -i python src/rag_generation.py
        ;;
    2)
        PYTHONPATH=. caffeinate -i python src/evaluate_rag.py
        ;;
    3)
        PYTHONPATH=. caffeinate -i python src/build_vectorstore.py
        ;;
    4)
        PYTHONPATH=. caffeinate -i python src/testset_generator.py
        ;;
    *)
        echo "Invalid option."
        ;;
esac

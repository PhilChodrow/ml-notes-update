clean: 
	rm -rf docs
	rm -rf _local-notes
	rm -rf index_files
	rm -rf _book

live-notebooks: 
	python scripts/create-ipynb.py

notes: 
	quarto render --profile private 

site: 
	quarto render --profile public

preview: 
	quarto preview --profile public
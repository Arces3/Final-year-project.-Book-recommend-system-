# Final-year-project.-Book-recommend-system-
## Project Overview
This project implements a **content-based Book Recommendation System** that provides personalized book suggestions using **TF-IDF vectorization** and **Cosine Similarity**. It also integrates a **multi-strategy fuzzy matching algorithm** to handle misspelled or partial queries. The system is deployed as an interactive **Streamlit web application**, enabling real-time book discovery.

---

## Features
- **Content-based recommendations** using book metadata: title, author, genre, and description.  
- **Cold-start solution**: works effectively for new books and new users.  
- **Multi-strategy fuzzy matching** for robust search:
  1. Exact match  
  2. Contains match  
  3. Word overlap  
  4. Smart fuzzy matching  
  5. Traditional fuzzy matching  
- **Real-time recommendations** with ultra-fast response (~0.003 seconds per query).  
- Multiple discovery options: search by title, author, genre, or explore the full dataset.

---

## Dataset
- Source: [Book Recommendation Dataset - Kaggle](https://www.kaggle.com/datasets)  
- Contains **10,059 books** with features:
  - Title  
  - Author  
  - Genre  
  - Description  
  - Rating  
- **Preprocessing steps**:
  - Cleaned missing values  
  - Normalized text (lowercase, removed punctuation)  
  - Standardized genres  
  - Removed duplicates  

---

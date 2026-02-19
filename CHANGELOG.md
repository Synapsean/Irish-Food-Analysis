# Changelog

All notable changes to the Irish Food Detective project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Database update script (`scripts/update_database.py`) for easy data refreshes
- Data quality report script (`scripts/data_quality_report.py`) for monitoring dataset health
- Maintenance guide (`MAINTENANCE.md`) with operational procedures
- Enhanced documentation on technical limitations and planned improvements

### Changed
- Updated README with clearer positioning of technical capabilities
- Added "Last Updated" badge to README

### Fixed
- Minor documentation typos

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- TF-IDF-based ingredient similarity recommender
- K-Means clustering for market segmentation
- Interactive Streamlit dashboard with 4 pages
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite with pytest
- Supabase database integration
- OpenFoodFacts API harvester

### Features
- Product recommender with healthier alternatives
- NOVA classification filtering
- Nutrient analysis across categories
- Statistical hypothesis testing
- ML clustering visualization

### Technical
- Full-stack data pipeline: API → Database → ML → Deployment
- Production deployment on Streamlit Cloud
- Automated testing and linting
- Modular code architecture

## Future Roadmap

### v1.1.0 (Q2 2026) - Semantic Similarity Upgrade
- [ ] Replace TF-IDF with Sentence-BERT embeddings
- [ ] Implement FAISS vector database for scalable similarity search
- [ ] Add semantic matching for ingredient synonyms
- [ ] Update evaluation metrics to use new embeddings

### v1.2.0 (Q3 2026) - Enhanced Evaluation
- [ ] Create expert-curated validation set (50+ products)
- [ ] Implement Precision@K, Recall@K, NDCG@K metrics
- [ ] Add A/B testing framework for recommendation quality
- [ ] Pharmacology-informed health scoring system

### v2.0.0 (Q4 2026) - Advanced Features
- [ ] Deep learning recommendation system (neural collaborative filtering)
- [ ] User preference learning
- [ ] Dietary restriction filters (vegan, gluten-free, allergen-free)
- [ ] Nutritional goal optimization

---

## Contributing

This is a personal portfolio project, but suggestions and feedback are welcome!
Please open an issue to discuss proposed changes.

## Contact

**Seán Quinlan, PhD**
- LinkedIn: [linkedin.com/in/sean-quinlan-phd](https://linkedin.com/in/sean-quinlan-phd)
- Email: sean.quinlan91@gmail.com
- GitHub: [github.com/Synapsean](https://github.com/Synapsean)

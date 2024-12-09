# ACS AI Query Tool

## The Power and Complexity of the American Community Survey (ACS)

For the full documentation, please visit the [project homepage.](https://michaelminzey.com/acs-query)

## [Access the ACS AI Query Tool here](https://victorious-stone-01080ed10.4.azurestaticapps.net/)


The American Community Survey (ACS) is a vital resource from the U.S. Census Bureau, providing detailed demographic, economic, and housing data. Its insights are invaluable for social and healthcare researchers, especially as social determinants of health—factors like income, education, and housing—are increasingly integrated into clinical studies.

Despite its value, the ACS dataset’s size—over 20,000 variables—makes it difficult to navigate. Existing tools, like the ACS API, require technical expertise, creating a barrier for non-technical users who could benefit from this data. Simplifying access to ACS data is essential to empower broader use in public health, social policy, and evidence-based decision-making.

---

## Simplifying ACS Data Access with AI-Powered Solutions

I’ve developed a solution to bridge the gap between the ACS’s complexity and the researchers who need its data. By leveraging artificial intelligence and large language models (LLMs), I’ve created an intuitive natural language interface that allows researchers to query the ACS without requiring technical expertise.

- **Natural Language Interface**: Users ask questions in plain language.
- **Dynamic API Construction**: The system discerns intent, identifies relevant API parameters, constructs API calls, retrieves data, and presents it in a user-friendly format.

---

## Example Query:

**Count of Service Workers Walking to Work in Michigan**  
ACS API Example: https://api.census.gov/data/2022/acs/acs5?get=NAME,B08124_031E&for=place:26

---

## Technologies and Architecture

- **Front-End**: Azure Static Web App, React, and Material UI for clean, intuitive user interfaces.
- **Back-End**: Azure Function (Python), serverless and scalable for query processing.
- **Database**: Azure SQL for storing reference data (API metadata, location codes, keyword dictionaries).
- **CI/CD**: GitHub CI/CD automates builds and testing, ensuring rapid iteration.

---

## Workflow

1. **Query Submission**: User submits a query through the interface.
2. **Intent Parsing**: LLM identifies the user’s intent, keywords, and location data.
3. **API Code Selection**: The system matches the query to API codes using categories and keywords.
4. **Data Retrieval**: Constructs API calls, retrieves data, and formats results for easy interpretation.
5. **Result Presentation**: Outputs user-friendly results, enabling actionable insights.

---

## Derived Categories

To narrow down the 20,000+ API variables, I grouped them into 13 categories:

1. **Education**: School enrollment, literacy rates.
2. **Disability**: Impact of disabilities on life and work.
3. **Population**: Demographics, age, gender.
4. **Transportation**: Commutes, travel times.
5. **Household and Family**: Family size, housing arrangements.
6. **Geographical Mobility**: Migration patterns.
7. **Race and Ancestry**: Cultural heritage.
8. **Employment**: Occupations, industries.
9. **Marriage and Birth**: Family trends.
10. **Language**: English proficiency.
11. **Poverty**: Economic hardship.
12. **Income**: Earnings, inequality.
13. **Age**: Age distributions, race, gender.

---

## Lessons Learned: Integrating LLMs

- **Structured Output**: Ensured JSON responses for integration into control logic.
- **Iterative Validation**: Used LLMs to validate results at each step, aligning outputs with user intent.
- **Error Handling**: Refined prompts to improve accuracy and usability.
- **Enhanced Labels**: Renamed lengthy API metric names for clarity.

---

## Final Thoughts and Future Enhancements

As a proof of concept, this solution effectively simplifies access to the ACS database, but there is room for improvement:

- **Expand Computation**: Moving to a robust architecture or queued process to handle complex queries.
- **Improve Semantic Matching**: Use advanced techniques for keyword interpretation.
- **Add Analytics**: Provide visualizations and comparison tools.
- **Handle Ambiguity**: Enable LLM interaction to clarify queries.
- **Export Results**: Support CSV/Excel formats for integration into workflows.

Maintaining accuracy is critical for a research tool, and future iterations will focus on reducing errors to ensure data integrity.

---

Thanks for reading!  
**- Michael**

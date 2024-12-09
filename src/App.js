import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Grid,
  Card,
  CardContent
} from "@mui/material";

function App() {
  const [userInput, setUserInput] = useState("");
  const [chatLog, setChatLog] = useState([]); // Stores all chat messages
  const [results, setResults] = useState([]); // Stores result data for display in the result section
  const [isLoading, setIsLoading] = useState(false); // Tracks API call status
  const chatEndRef = useRef(null); // Ref for the bottom of the chat log

// Scroll to the bottom of the chat window when chatLog changes
useEffect(() => {
  if (chatEndRef.current) {
    chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }
}, [chatLog]);

const handleSubmit = async (e) => {
  e.preventDefault();

  // Clear user input
  setUserInput("");

  // Add user input to chat log
  setChatLog([...chatLog, { role: "user", content: userInput }]);
  setIsLoading(true); // Show spinner

  try {
    // Call the Azure Function API
    const response = await fetch("/api/query", {  
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: userInput }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.status === "success" && Array.isArray(data.data)) {
      // Add assistant response to chat log
      setChatLog((prev) => [
        ...prev,
        { role: "assistant", content: "Here are the results for your query." },
      ]);

      // Prepend new results (with title) to the results array
      setResults((prev) => [
        { title: data.title || "Result", data: data.data }, // Use API-provided title or fallback
        ...prev,
      ]);
    } else {
      setChatLog((prev) => [
        ...prev,
        { role: "assistant", content: data.message },
      ]);
    }
  } catch (error) {
    console.error("Error calling Azure Function:", error);
  
    let errorMessage = "Oof, sorry I ran out of time and Azure told me to stop. Maybe try something more specific?";
  
    if (error.response && error.response.data && error.response.data.message) {
      errorMessage = error.response.data.message; // Use API-provided message if available
    }
  
    setChatLog((prev) => [
      ...prev,
      { role: "assistant", content: errorMessage },
    ]);
  } finally {
    setIsLoading(false); // Hide spinner
  }
};


  return (
    <Box
    sx={{
      width: "100%",
      minHeight: "100vh",
      background: "linear-gradient(to bottom, #2980b9, #d6eaf8)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
    }}
  >
    <Box
      sx={{
        maxWidth: "800px",
        margin: "0 auto",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",

      }}
    >
      {/* Static Content Section */}
      <Paper elevation={3} sx={{ padding: "20px" }}>
        <Typography variant="h4" gutterBottom>
          ACS AI Query Tool
        </Typography>
        <Typography variant="body1" gutterBottom sx={{ marginBottom: "20px" }}>
          The American Community Survey (ACS) is a vital resource provided by the U.S. Census Bureau. It offers comprehensive data on the social, economic, housing, and demographic characteristics of communities across the United States. The ACS helps policymakers, businesses, researchers, and the public make informed decisions by providing insights into areas like education, income, employment, housing trends, and much more.
        </Typography>
        <Typography variant="body2" gutterBottom sx={{ marginBottom: "20px" }}>
        <a href="https://michaelminzey.com/acs-query" target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none", color: "#1976d2" }}>
          Learn more about this tool and how it was built.
        </a>
      </Typography>
        <Typography variant="h5" gutterBottom>
          What You Can Do with This Tool:
        </Typography>
        <Typography variant="body1" gutterBottom sx={{ marginBottom: "20px" }}>
          This tool allows you to interact with ACS data through natural language queries. Simply type your question in the chat window, and the assistant will analyze your request, retrieve the relevant data, and display the results. The tool is designed to make accessing complex datasets simple and intuitive.
        </Typography>
        <Typography variant="h5" gutterBottom>
          Examples of Questions You Can Ask:
        </Typography>
        <Typography variant="body2" gutterBottom>
          1. <strong>"What is the median income in Grand Rapids, MI?"</strong>  
        </Typography>
        <Typography variant="body2" gutterBottom>
          2. <strong>"How many people commute by public transportation in New York City?"</strong>  
        </Typography>
        <Typography variant="body2" gutterBottom sx={{ marginBottom: "20px" }}>
          3. <strong>"Compare the Hispanic population of California and Texas."</strong>  
        </Typography>
        <Typography variant="h5" gutterBottom>
          Tips for Using the Tool:
        </Typography>
        <Typography variant="body1" gutterBottom>
          - Include <strong>specific locations</strong> like states, counties, or cities in your query to refine the results.  
            Example: <em>"How many people are over 18 years old in Detroit, Michigan?"</em>
        </Typography>
        <Typography variant="body1" gutterBottom>
          - Be clear about the <strong>type of information</strong> you're requesting, such as income, education, or employment.  
            Example: <em>"How many people are hearing and vision disabled in Florida?"</em>
        </Typography>
        <Typography variant="body1" gutterBottom sx={{ marginBottom: "20px" }}>
          - For comparisons, mention all relevant locations or categories.  
            Example: <em>"Compare household incomes between Ohio and Pennsylvania."</em>
        </Typography>
        <Typography variant="body1" gutterBottom>
          Feel free to explore and ask your questions below. The chat log will record your interactions, and results will be displayed dynamically. Happy querying!
        </Typography>
      </Paper>
      <Paper elevation={3} sx={{ padding: "20px", marginTop: "20px" }}>
  <Typography variant="h5" gutterBottom>
    ACS API Categories
  </Typography>
  <Grid container spacing={2}>
    {[
      { title: "Education", description: "Educational attainment, school enrollment, and literacy rates." },
      { title: "Disability", description: "Data on individuals with disabilities and their impact on daily life or employment." },
      { title: "Population", description: "Total population counts, age, gender, and density." },
      { title: "Transportation", description: "Commuting patterns, travel times, and vehicle ownership." },
      { title: "Household and Family", description: "Family size, household types, and housing arrangements." },
      { title: "Geographical Mobility", description: "Movement between geographic locations and migration patterns." },
      { title: "Race and Ancestry", description: "Racial and ethnic groups, ancestry, and cultural heritage." },
      { title: "Employment", description: "Employment status, industries, and occupations." },
      { title: "Marriage and Birth", description: "Marital status, births, and family formation trends." },
      { title: "Language", description: "Languages spoken at home and English proficiency." },
      { title: "Poverty", description: "Individuals living below the poverty line and economic hardship indicators." },
      { title: "Income", description: "Earnings, household income, and income inequality." },
      { title: "Age", description: "Age distribution of the population, including race and gender." },
    ].map((category, index) => (
      <Grid item xs={12} sm={6} md={4} key={index}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {category.title}
            </Typography>
            <Typography variant="body2">{category.description}</Typography>
          </CardContent>
        </Card>
      </Grid>
    ))}
  </Grid>
</Paper>

      {/* Chat Window Section */}
      <Paper elevation={3} sx={{ padding: "20px" }}>
        <Typography variant="h5" gutterBottom>
          Chat Window
        </Typography>
        <Box
          sx={{
            maxHeight: "300px",
            overflowY: "auto",
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            marginBottom: "10px",
          }}
        >
          {chatLog.map((entry, index) => (
            <Typography
              key={index}
              variant="body2"
              sx={{
                textAlign: entry.role === "user" ? "right" : "left",
                color: entry.role === "user" ? "blue" : "green",
                marginBottom: "5px",
                display: "block", // Ensures messages are block elements for alignment
              }}
            >
              <strong>{entry.role === "user" ? "You:" : "Assistant:"}</strong>{" "}
              {entry.content}
            </Typography>
          ))}
          {/* Add a hidden div to track the bottom of the chat */}
          <div ref={chatEndRef}></div>
        </Box>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            variant="outlined"
            label="Enter your question"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            sx={{ marginBottom: "10px" }}
          />
          <Button variant="contained" type="submit" fullWidth>
            Submit
          </Button>
        </form>
      </Paper>

      {/* Spinner Section */}
      {isLoading && (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            marginTop: "20px",
          }}
        >
          <CircularProgress />
          <Typography variant="body2" sx={{ marginTop: "10px" }}>
            Processing...
          </Typography>
        </Box>
      )}

      {/* Result Section */}
      <Paper elevation={3} sx={{ padding: "20px" }}>
        <Typography variant="h5" gutterBottom>
          Results
        </Typography>
        {results.length > 0 ? (
          results.map((resultSet, resultIndex) => (
            <Box key={resultIndex} sx={{ marginBottom: "20px" }}>
              <Typography variant="h6">{resultSet.title}</Typography> {/* Display result title */}
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      {Object.keys(resultSet.data[0]).map((key) => (
                        <TableCell key={key}>{key}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {resultSet.data.map((row, rowIndex) => (
                      <TableRow key={rowIndex}>
                        {Object.values(row).map((value, cellIndex) => (
                          <TableCell key={cellIndex}>{value}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          ))
        ) : (
          <Typography>No results to display.</Typography>
        )}
      </Paper>
    </Box>
    </Box>
  );
}

export default App;

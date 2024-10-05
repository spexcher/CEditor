import { Box } from "@chakra-ui/react";
import CodeEditor from "./components/CodeEditor";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import SnippetPage from './components/SnippetPage';

function App() {
  return (
    <Box minH="100vh" bg="#0f0a19" color="gray.500" px={6} py={8}>
      <Router>
        <Routes>
          {/* Route to CodeEditor component */}
          <Route path="/" element={<CodeEditor />} />
          {/* Route to SnippetPage component */}
          <Route path="/snippets" element={<SnippetPage />} />
        </Routes>
      </Router>
    </Box>
  );
}

export default App;

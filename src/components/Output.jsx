import { useState } from "react";
import { Box, Button, Text, useToast, Textarea } from "@chakra-ui/react";
import { executeCode } from "../api";

const Output = ({ editorRef, language }) => {
  // Receive input as a prop
  const toast = useToast();
  const [output, setOutput] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [input, setInput] = useState("");
  const runCode = async () => {
    const sourceCode = editorRef.current.getValue();
    // if (!input) return;  // Check for input
    if (!sourceCode) return;
    try {
      setIsLoading(true);
      const { run: result } = await executeCode(language, sourceCode, input); // Pass input to the API
      setOutput(result.output.split("\n"));
      result.stderr ? setIsError(true) : setIsError(false);
    } catch (error) {
      console.log(error);
      toast({
        title: "An error occurred.",
        description: error.message || "Unable to run code",
        status: "error",
        duration: 6000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Box w="40%">
        <Button
          variant="outline"
          colorScheme="red"
          mb={4}
          isLoading={isLoading}
          onClick={runCode}
          p={8}
          fontSize={50}
          ml={200}
        >
          ▶ Run
        </Button>
        <Text mb={2} fontSize="lg">
          Input
        </Text>
        <Textarea
          //height="20vh"
          height="30vh"
          mt={2}
          //p={2}
          color="#fff"
          border="1px solid"
          borderRadius={4}
          borderColor="#333"
          overflow="auto"
          placeholder="Enter input (If any)"
          textDecoration="none"
          onChange={(e) => setInput(e.target.value)}
        ></Textarea>
        <Text mb={2} fontSize="lg">
          Output
        </Text>
        <Box
          height="50vh"
          p={2}
          color={isError ? "red.400" : ""}
          border="1px solid"
          borderRadius={4}
          borderColor={isError ? "red.500" : "#333"}
          overflow="auto"
        >
          {output
            ? output.map((line, i) => <Text key={i}>{line}</Text>)
            : 'Click "Run Code" to see the output here'}
        </Box>
      </Box>
    </>
  );
};

export default Output;

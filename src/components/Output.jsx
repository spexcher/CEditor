import { useState } from "react";
import { Box, Button, Text, useToast, Textarea, VStack } from "@chakra-ui/react";
import { executeCode } from "../api";

const Output = ({ editorRef, language, scrollRef }) => {
  const toast = useToast();
  const [output, setOutput] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [input, setInput] = useState("");

  const runCode = async () => {
    const sourceCode = editorRef.current.getValue();
    if (!sourceCode) return;
    if (window.innerWidth < 768 && scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }

    try {
      setIsLoading(true);
      const { run: result } = await executeCode(language, sourceCode, input);
      setOutput(result.output.split("\n"));
      setIsError(!!result.stderr);
    } catch (error) {
      toast({
        title: "Error",
        description: error.message || "Unable to run code",
        status: "error",
        duration: 4000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <VStack align="stretch" spacing={4}>
      <Button
        variant="solid"
        colorScheme="blue" 
        isLoading={isLoading}
        onClick={runCode}
        size="lg"
        width="100%"
        height="60px"
        fontSize="xl"
        borderRadius="xl"
      >
        ▶ Run Code
      </Button>

      <Box>
        <Text mb={2} fontWeight="bold" fontSize="sm" color="gray.400">INPUT (OPTIONAL)</Text>
        <Textarea
          height="120px"
          bg="#0f0f0f"
          borderColor="#333"
          placeholder="Enter input here..."
          onChange={(e) => setInput(e.target.value)}
        />
      </Box>

      <Box>
        <Text mb={2} fontWeight="bold" fontSize="sm" color="gray.400">CONSOLE OUTPUT</Text>
        <Box
          height={{ base: "300px", md: "50vh" }}
          p={3}
          color={isError ? "red.400" : "green.200"}
          bg="#0f0f0f"
          border="1px solid"
          borderColor={isError ? "red.800" : "#333"}
          borderRadius="md"
          overflow="auto"
          fontFamily="monospace"
          fontSize="sm"
        >
          {output
            ? output.map((line, i) => <Text key={i}>{line}</Text>)
            : 'Terminal ready. Click "Run Code" to execute.'}
        </Box>
      </Box>
    </VStack>
  );
};

export default Output;

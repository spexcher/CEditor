import { useRef, useState, useEffect } from "react";
import { Box, Stack, HStack, Button, Link, Select, Icon, Flex, Wrap, WrapItem } from "@chakra-ui/react";
import { Editor } from "@monaco-editor/react";
import LanguageSelector from "./LanguageSelector";
import { CODE_SNIPPETS } from "../constants";
import Output from "./Output";
import * as monaco from "monaco-editor";
import { FaGithub, FaLinkedin, FaFacebook, FaInstagram } from "react-icons/fa";
import { SiCodeforces, SiLeetcode, SiCodechef } from "react-icons/si";

const CodeEditor = () => {
  const editorRef = useRef();
  const outputSectionRef = useRef(null); // Ref to scroll to output
  const [value, setValue] = useState("");
  const [language, setLanguage] = useState("cpp");
  const [theme, setTheme] = useState("vs-dark");

  const onMount = (editor) => {
    editorRef.current = editor;
    editor.focus();
  };

  const onSelect = (language) => {
    setLanguage(language);
    setValue(CODE_SNIPPETS[language]);
  };

  const handleThemeChange = (e) => {
    const selectedTheme = e.target.value;
    setTheme(selectedTheme);
    monaco.editor.setTheme(selectedTheme);
  };

  useEffect(() => {
    monaco.editor.setTheme(theme);
  }, [theme]);

  return (
    <Stack direction={{ base: "column", md: "row" }} spacing={6} p={{ base: 2, md: 6 }} align="flex-start">
      <Box w={{ base: "100%", md: "60%" }}>
        {/* Responsive Header Controls */}
        <Flex direction="column" gap={3} mb={4}>
          <HStack justify="space-between" wrap="wrap">
            <LanguageSelector language={language} onSelect={onSelect} />
            <Link href="/snippets" isExternal>
              <Button colorScheme="red" size="sm">Snippets</Button>
            </Link>
          </HStack>

          <Wrap spacing={2} justify={{ base: "center", md: "flex-start" }}>
            <Link href="https://github.com/spexcher/CEdItor" isExternal>
              <Button colorScheme="green" size="xs" leftIcon={<Icon as={FaGithub} />}>
                <Box as="span" display={{ base: "none", sm: "inline" }}>Star on GitHub</Box>
              </Button>
            </Link>
            <HStack spacing={3} p={1.5} border="1px solid #38A169" borderRadius="md">
                <Link href="https://github.com/spexcher" isExternal><Icon as={FaGithub} color="#9AE6B4" /></Link>
                <Link href="https://www.linkedin.com/in/gourabmodak/" isExternal><Icon as={FaLinkedin} color="#9AE6B4" /></Link>
                <Link href="https://leetcode.com/spexcher/" isExternal><Icon as={SiLeetcode} color="#9AE6B4" /></Link>
            </HStack>
          </Wrap>
        </Flex>

        <Select onChange={handleThemeChange} value={theme} mb={4} size="sm">
          <option value="vs">Light Theme</option>
          <option value="vs-dark">Dark Theme</option>
          <option value="hc-black">High Contrast</option>
        </Select>

        {/* Editor Wrapper with guaranteed height */}
        <Box 
          height={{ base: "60vh", md: "75vh" }} 
          border="1px solid #333" 
          borderRadius="md" 
          overflow="hidden"
        >
          <Editor
            options={{
              minimap: { enabled: false },
              automaticLayout: true,
              wordWrap: "on",
              padding: { top: 10 },
            }}
            theme={theme}
            language={language}
            defaultValue={CODE_SNIPPETS[language]}
            onMount={onMount}
            value={value}
            onChange={(val) => setValue(val)}
          />
        </Box>
      </Box>

      {/* Output Section */}
      <Box w={{ base: "100%", md: "40%" }} ref={outputSectionRef}>
        <Output editorRef={editorRef} language={language} scrollRef={outputSectionRef} />
      </Box>
    </Stack>
  );
};

export default CodeEditor;

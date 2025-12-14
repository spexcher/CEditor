
import { useRef, useState, useEffect } from "react";
import {
  Box,
  Stack,
  HStack,
  Button,
  Link,
  Select,
  Icon,
  Flex,
  Wrap,
  Text,
  Tooltip,
  useColorModeValue,
} from "@chakra-ui/react";
import { FaGithub, FaLinkedin, FaFacebook, FaInstagram } from "react-icons/fa";
import { SiCodechef, SiCodeforces, SiLeetcode } from "react-icons/si";
import { Editor } from "@monaco-editor/react";
import LanguageSelector from "./LanguageSelector";
import { CODE_SNIPPETS } from "../constants";
import Output from "./Output";
import * as monaco from "monaco-editor";

const CodeEditor = () => {
  const editorRef = useRef(null);
  const outputSectionRef = useRef(null);

  const [value, setValue] = useState("");
  const [language, setLanguage] = useState("cpp");
  const [theme, setTheme] = useState("vs-dark");

  const socialBg = useColorModeValue(
    "linear(to-r, green.300, green.400)",
    "linear(to-r, green.500, green.600)"
  );

  const onMount = (editor) => {
    editorRef.current = editor;
    editor.focus();
  };

  const onSelect = (lang) => {
    setLanguage(lang);
    setValue(CODE_SNIPPETS[lang]);
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
    <Stack
      direction={{ base: "column", md: "row" }}
      spacing={6}
      p={{ base: 3, md: 6 }}
      align="flex-start"
                Snippets
    >
      <Box w={{ base: "100%", md: "60%" }}>

        <Flex direction="column" gap={4} mb={4}>
          <HStack justify="space-between" wrap="wrap">
            <LanguageSelector language={language} onSelect={onSelect} />

            <Link href="/snippets" isExternal>
              <Button size="lg" colorScheme="red" fontWeight="bold">
              </Button>
            </Link>
          </HStack>

    
          <Wrap spacing={4} justify={{ base: "center", md: "flex-start" }}>

            <Link href="https://github.com/spexcher/CEdItor" isExternal>
              <Button
                leftIcon={<Icon as={FaGithub} boxSize={5} />}
                size="sm"
                px={4}
                py={2}
                borderRadius="lg"
                fontWeight="bold"
                bgGradient="linear(to-r, blue.300, blue.400)"
                color="black"
                boxShadow="md"
                _hover={{
                  bgGradient: "linear(to-r, blue.400, blue.500)",
                  color: "white",
                }}
                _active={{ transform: "scale(0.97)" }}
              >
                Star on GitHub
              </Button>
            </Link>

            <HStack
              spacing={5}
              p={4}
              wrap="wrap"
              justify="center"
              borderRadius="xl"
              // bgGradient={socialBg}
              bgGradient="linear(to-r, blue.300, blue.400)"
              boxShadow="lg"
            >
              <Text fontWeight="bold" fontSize="lg" color="#0d140dff">
                Find me here
              </Text>

              {[
                {
                  icon: FaGithub,
                  label: "GitHub",
                  link: "https://github.com/spexcher",
                },
                {
                  icon: SiCodechef,
                  label: "CodeChef",
                  link: "https://www.codechef.com/users/spexcher",
                },
                {
                  icon: FaLinkedin,
                  label: "LinkedIn",
                  link: "https://www.linkedin.com/in/gourabmodak/",
                },
                {
                  icon: SiCodeforces,
                  label: "Codeforces",
                  link: "https://codeforces.com/profile/spexcher",
                },
                {
                  icon: SiLeetcode,
                  label: "LeetCode",
                  link: "https://leetcode.com/spexcher/",
                },
                {
                  icon: FaFacebook,
                  label: "Facebook",
                  link: "https://facebook.com/spexcher",
                },
                {
                  icon: FaInstagram,
                  label: "Instagram",
                  link: "https://instagram.com/spexcher",
                },
              ].map(({ icon, label, link }) => (
                <Tooltip label={label} key={label} hasArrow>
                  <Link href={link} isExternal>
                    <Icon
                      as={icon}
                      boxSize={8}
                      color="#000000"
                      transition="color 0.2s ease"
                      _hover={{ color: "white" }}
                    />
                  </Link>
                </Tooltip>
              ))}
            </HStack>
          </Wrap>
        </Flex>
        <Select
          onChange={handleThemeChange}
          value={theme}
          mb={4}
          size="sm"
          maxW="200px"
        >
          <option value="vs">Light Theme</option>
          <option value="vs-dark">Dark Theme</option>
          <option value="hc-black">High Contrast</option>
        </Select>
        <Box
          height={{ base: "60vh", md: "75vh" }}
          border="1px solid"
          borderColor="gray.700"
          borderRadius="lg"
          overflow="hidden"
          boxShadow="xl"
        >
          <Editor
            options={{
              minimap: { enabled: false },
              automaticLayout: true,
              wordWrap: "on",
              padding: { top: 12 },
              fontSize: 14,
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
      <Box w={{ base: "100%", md: "40%" }} ref={outputSectionRef}>
        <Output
          editorRef={editorRef}
          language={language}
          scrollRef={outputSectionRef}
        />
      </Box>
    </Stack>
  );
};

export default CodeEditor;

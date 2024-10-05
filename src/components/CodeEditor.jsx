import { useRef, useState, useEffect } from "react";
import { Box, HStack, Button, Link, Select, Icon } from "@chakra-ui/react";
import { Editor } from "@monaco-editor/react";
import LanguageSelector from "./LanguageSelector";
import { CODE_SNIPPETS } from "../constants";
import Output from "./Output";
import * as monaco from "monaco-editor";
import { FaGithub, FaLinkedin, FaFacebook, FaInstagram } from "react-icons/fa";
import { SiCodeforces, SiLeetcode } from "react-icons/si";
import { SiCodechef } from "react-icons/si";
const CodeEditor = () => {
  const editorRef = useRef();
  const [value, setValue] = useState("");
  const [language, setLanguage] = useState("cpp");

  const [theme, setTheme] = useState("vs-dark"); // Default theme

  const onMount = (editor) => {
    editorRef.current = editor;
    editor.focus();
  };

  const onSelect = (language) => {
    setLanguage(language);
    setValue(CODE_SNIPPETS[language]);
  };

  const handleThemeChange = (event) => {
    const selectedTheme = event.target.value;
    setTheme(selectedTheme);
    monaco.editor.setTheme(selectedTheme); // Set the theme directly
  };

  const handleFormatCode = () => {
    if (editorRef.current) {
      // Trigger the formatting action
      editorRef.current.getAction("editor.action.formatDocument").run();
      console.log("FOrmatting Done");
    }
  };

  useEffect(() => {
    // Load the selected theme
    const loadTheme = async (theme) => {
      monaco.editor.setTheme(theme);
    };

    loadTheme(theme); // Apply the initial theme on mount
  }, [theme]);

  return (
    <HStack spacing={4}>
      <Box w="60%">
        <HStack mb={4}>
          <LanguageSelector language={language} onSelect={onSelect} />
          <Link href="/snippets" isExternal>
            <Button mt={5} colorScheme="red">
              Snippets
            </Button>
          </Link>
          <Link href="https://github.com/spexcher/CEdItor" isExternal>
            <Button
              mt={5}
              colorScheme="green"
              size="lg"
              rightIcon={<Icon as={FaGithub} />}
              _hover={{ bg: "green.500" }}
            >
              🫶 Support by giving a ⭐️ on
            </Button>
          </Link>
          <div
            style={{ display: "flex", gap: "20px", "margin-top": "1.2rem",color:"#9AE6B4",border:"1px solid #38A169",padding: "1rem",borderRadius:"0.5rem"}}
          >
            <Link
              href="https://github.com/spexcher"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaGithub size={30} />
            </Link>
            <Link
              href="https://www.codechef.com/users/spexcher"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiCodechef size={30} />
            </Link>
            <Link
              href="https://www.linkedin.com/in/gourabmodak/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaLinkedin size={30} />
            </Link>
            <Link
              href="https://codeforces.com/profile/spexcher"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiCodeforces size={30} />
            </Link>
            <Link
              href="https://leetcode.com/spexcher/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiLeetcode size={30} />
            </Link>
            <Link
              href="https://facebook.com/spexcher"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaFacebook size={30} />
            </Link>
            <Link
              href="https://instagram.com/spexcher"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FaInstagram size={30} />
            </Link>
          </div>
          {/* Format Button */}
          {/* <Button mt={5} colorScheme="blue" onClick={handleFormatCode}>
            Format Code
          </Button> */}
        </HStack>

        <Select onChange={handleThemeChange} value={theme} mb={4}>
          <option value="vs">Light Theme</option>
          <option value="vs-dark">Dark Theme</option>
          <option value="hc-black">High Contrast Black Theme</option>
        </Select>

        <Editor
          options={{
            minimap: {
              enabled: true,
            },
            automaticLayout: true,
            formatOnType: true,
            formatOnPaste: true,
            wordWrap: "on",
            autoIndent: "full",
            tabSize: 4,
          }}
          height="90vh"
          theme={theme}
          language={language}
          defaultValue={CODE_SNIPPETS[language]}
          onMount={onMount}
          value={value}
          onChange={(value) => setValue(value)}
        />
      </Box>

      <Output editorRef={editorRef} language={language} />
    </HStack>
  );
};

export default CodeEditor;

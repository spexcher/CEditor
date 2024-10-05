// import { useRef, useState, useEffect } from "react";
// import {
//   Box,
//   HStack,
//   Textarea,
//   Text,
//   Button,
//   Link,
//   Select,
//   VStack,
// } from "@chakra-ui/react";
// import { Editor } from "@monaco-editor/react";
// import LanguageSelector from "./LanguageSelector";
// import { CODE_SNIPPETS } from "../constants";
// import Output from "./Output";
// import * as monaco from "monaco-editor";
// import * as monacothemes from "monaco-themes"; // Import monaco-themes

// const CodeEditor = () => {
//   const editorRef = useRef();
//   const [value, setValue] = useState("");
//   const [language, setLanguage] = useState("cpp");

//   const [theme, setTheme] = useState("vs-dark"); // Default theme

//   const onMount = (editor) => {
//     editorRef.current = editor;
//     editor.focus();
//   };

//   const onSelect = (language) => {
//     setLanguage(language);
//     setValue(CODE_SNIPPETS[language]);
//   };

//   const handleThemeChange = (event) => {
//     const selectedTheme = event.target.value;
//     setTheme(selectedTheme);
//     monaco.editor.setTheme(selectedTheme); // Set the theme directly
//   };

//   useEffect(() => {
//     // Load the selected theme
//     const loadTheme = async (theme) => {
//       // const themeData = await monacothemes.getTheme(theme);
//       //monaco.editor.defineTheme(theme, themeData);
//       monaco.editor.setTheme(theme);
//     };

//     loadTheme(theme); // Apply the initial theme on mount
//   }, [theme]);

//   return (
//     <HStack spacing={4}>
//       <Box w="60%">
//         <HStack mb={4}>
//           <LanguageSelector language={language} onSelect={onSelect} />
//           <Link href="/snippets" isExternal>
//             <Button mt={5} colorScheme="red">
//               Snippets
//             </Button>
//           </Link>
//         </HStack>

//         <Select onChange={handleThemeChange} value={theme} mb={4}>
//           <option value="vs">Light Theme</option>
//           <option value="vs-dark">Dark Theme</option>
//           <option value="hc-black">High Contrast Black Theme</option>
//         </Select>

//         <Editor
//           options={{
//             minimap: {
//               enabled: true,
//             },
//             automaticLayout: true, // Ensures the editor resizes automatically
//             formatOnType: true, // Format code as you type
//             formatOnPaste: true, // Format code when pasting
//             wordWrap: "on", // Wrap text
//             autoIndent: "full", // Ensures full auto-indent support
//             tabSize: 4, // Set tab size (optional, can be adjusted)
//             insertSpaces: true, // Insert spaces instead of tabs (optional)
//           }}
//           height="90vh"
//           theme={theme}
//           language={language}
//           defaultValue={CODE_SNIPPETS[language]}
//           onMount={onMount}
//           value={value}
//           onChange={(value) => setValue(value)}
//         />

//         {/* <Textarea
//           mt={4}
//           height="15vh"
//           placeholder="Enter input for the program"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//         /> */}
//       </Box>

//       <Output editorRef={editorRef} language={language} />
//     </HStack>
//   );
// };

// export default CodeEditor;

import { useRef, useState, useEffect } from "react";
import { Box, HStack, Button, Link, Select, Icon } from "@chakra-ui/react";
import { Editor } from "@monaco-editor/react";
import LanguageSelector from "./LanguageSelector";
import { CODE_SNIPPETS } from "../constants";
import Output from "./Output";
import * as monaco from "monaco-editor";
import { FaGithub } from "react-icons/fa";
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

import {
  Box,
  Heading,
  Text,
  VStack,
  Code,
  Divider,
  Button,
  useClipboard,
  HStack,
} from "@chakra-ui/react";
import { FaGithub, FaLinkedin, FaFacebook, FaInstagram } from "react-icons/fa";
import { SiCodeforces, SiLeetcode } from "react-icons/si";
import { SiCodechef } from "react-icons/si";
import { Link } from "react-router-dom";
import segmentTreeCode from "./cpp_snippets/Segtree_cpp";
import hld_code from "./cpp_snippets/HLD";
import add_multiply_strings_code from "./cpp_snippets/AM_strings";
import { combinatorics_code } from "./cpp_snippets/Combinatorics";

import { simple_segment_tree_code } from "./cpp_snippets/Algos";
import { matrix_code } from "./cpp_snippets/Algos";
import { binarySearchCode } from "./cpp_snippets/Algos";
import { dijkstraCode } from "./cpp_snippets/Algos";
import { substring_hash_code } from "./cpp_snippets/Algos";
import { DSU_code } from "./cpp_snippets/Algos";
import { multiply_polynomials_code } from "./cpp_snippets/Algos";
import { KMP_code } from "./cpp_snippets/Algos";
import { ETF_code } from "./cpp_snippets/Algos";
import { SOE_code } from "./cpp_snippets/Algos";
const SnippetPage = () => {
  // Clipboard hooks
  const { hasCopied: hasCopiedSegmentTree, onCopy: onCopySegmentTree } =
    useClipboard(segmentTreeCode);
  const {
    hasCopied: hasCopiedCombinatorics_code,
    onCopy: onCopyCombinatorics_code,
  } = useClipboard(combinatorics_code);
  const { hasCopied: hasCopiedBinarySearch, onCopy: onCopyBinarySearch } =
    useClipboard(binarySearchCode);
  const { hasCopied: hasCopiedDijkstra, onCopy: onCopyDijkstra } =
    useClipboard(dijkstraCode);
  const { hasCopied: hasCopiedMatrix, onCopy: onCopyMatrix } =
    useClipboard(matrix_code);
  const { hasCopied: hasCopiedDSU, onCopy: onCopyDSU } = useClipboard(DSU_code);
  const {
    hasCopied: hasCopied_substring_hash_code,
    onCopy: onCopy_substring_hash_code,
  } = useClipboard(substring_hash_code);
  const { hasCopied: hasCopied_hld_code, onCopy: onCopy_hld_code } =
    useClipboard(substring_hash_code);
  const {
    hasCopied: hasCopied_add_multiply_strings,
    onCopy: onCopy_add_multiply_strings,
  } = useClipboard(add_multiply_strings_code);
  const {
    hasCopied: hasCopied_multiply_polynomials,
    onCopy: onCopy_multiply_polynomials,
  } = useClipboard(multiply_polynomials_code);
  const { hasCopied: hasCopied_KMP, onCopy: onCopy_KMP } =
    useClipboard(KMP_code);
  const { hasCopied: hasCopied_ETF, onCopy: onCopy_ETF } =
    useClipboard(ETF_code);
  const { hasCopied: hasCopied_SOE, onCopy: onCopy_SOE } =
    useClipboard(SOE_code);
  const {
    hasCopied: hasCopied_simple_segment_tree,
    onCopy: onCopy_simple_segment_tree,
  } = useClipboard(simple_segment_tree_code);

  return (
    <Box p={8}>
      <Heading as="h1" size="xl" mb={2}>
        Competitive Programming Snippets by spexcher*
      </Heading>
      
      <Heading as="h2" size="ml" mb={2}>
        Find me at
      </Heading>

      <div
        style={{
          display: "flex",
          gap: "20px",
          //"margin-top": "0.2rem",
          color: "#9AE6B4",
          padding: "1rem",
          borderRadius: "0.5rem",
        }}
      >
        <a
          href="https://github.com/spexcher"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaGithub size={30} />
        </a>
        <a
          href="https://www.codechef.com/users/spexcher"
          target="_blank"
          rel="noopener noreferrer"
        >
          <SiCodechef size={30} />
        </a>
        <a
          href="https://www.linkedin.com/in/gourabmodak/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaLinkedin size={30} />
        </a>
        <a
          href="https://codeforces.com/profile/spexcher"
          target="_blank"
          rel="noopener noreferrer"
        >
          <SiCodeforces size={30} />
        </a>
        <a
          href="https://leetcode.com/spexcher/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <SiLeetcode size={30} />
        </a>
        <a
          href="https://facebook.com/spexcher"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaFacebook size={30} />
        </a>
        <a
          href="https://instagram.com/spexcher"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaInstagram size={30} />
        </a>
      </div>
      <Divider />
      <Heading as="h2" size="ml" mb={2}>Please Use CTRL/CMD + F to find your required Snippet HaHa.</Heading>
      <Divider />
      <VStack align="start" spacing={8}>
        {/* Simple Segment Tree */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Simple Segment Tree
            </Heading>
            <Button size="sm" onClick={onCopy_simple_segment_tree}>
              {hasCopied_simple_segment_tree ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            A segment tree is a data structure that allows querying and updating
            ranges of an array efficiently.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {simple_segment_tree_code}
          </Code>
        </Box>
        <Divider />
        {/* Segment Tree */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Segment Tree
            </Heading>
            <Button size="sm" onClick={onCopySegmentTree}>
              {hasCopiedSegmentTree ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            A segment tree is a data structure that allows querying and updating
            ranges of an array efficiently.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {segmentTreeCode}
          </Code>
        </Box>
        <Divider />


        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
             For Factorials and Cominatorics questions with modular integration
            </Heading>
            <Button size="sm" onClick={onCopyCombinatorics_code}>
              {hasCopiedCombinatorics_code ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            nCr nPr factorials modular inverses and much more..
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {combinatorics_code}
          </Code>
        </Box>

        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              DSU (Disjoint Set Union / Union Find Data Structure)
            </Heading>
            <Button size="sm" onClick={onCopyDSU}>
              {hasCopiedDSU ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            The Disjoint Set Union (DSU) data structure, which allows you to add
            edges to a graph and test whether two vertices of the graph are
            connected.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {DSU_code}
          </Code>
        </Box>

        <Divider />

        {/* Binary Search */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Binary Search
            </Heading>
            <Button size="sm" onClick={onCopyBinarySearch}>
              {hasCopiedBinarySearch ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Binary search is a classic algorithm for finding an element in a
            sorted array.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {binarySearchCode}
          </Code>
        </Box>

        <Divider />

        {/* Substring Hash Code */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Substring Hash Code
            </Heading>
            <Button size="sm" onClick={onCopy_substring_hash_code}>
              {hasCopied_substring_hash_code ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>Substring Hash in O(1)</Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {substring_hash_code}
          </Code>
        </Box>

        <Divider />

        {/* Dijkstra's Algorithm */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Dijkstra's Algorithm
            </Heading>
            <Button size="sm" onClick={onCopyDijkstra}>
              {hasCopiedDijkstra ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            Dijkstra's algorithm finds the shortest path between nodes in a
            graph.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {dijkstraCode}
          </Code>
        </Box>
        <Divider />

        {/* Matrix Operations */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Matrix Operations
            </Heading>
            <Button size="sm" onClick={onCopyMatrix}>
              {hasCopiedMatrix ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>Common Matrix operations and exponentiation</Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {matrix_code}
          </Code>
        </Box>
        <Divider />
        {/* Add, Multiply Strings */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Add, Multiply Strings
            </Heading>
            <Button size="sm" onClick={onCopy_add_multiply_strings}>
              {hasCopied_add_multiply_strings ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>Functions to Add and Multiply Strings</Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {add_multiply_strings_code}
          </Code>
        </Box>
        <Divider />
        {/* Multiply Polynomials */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Multiply Polynomials
            </Heading>
            <Button size="sm" onClick={onCopy_multiply_polynomials}>
              {hasCopied_multiply_polynomials ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>Multiply Polynomials by Karastuba Method</Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {multiply_polynomials_code}
          </Code>
        </Box>
        <Divider />
        {/* KMP */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              KMP
            </Heading>
            <Button size="sm" onClick={onCopy_KMP}>
              {hasCopied_KMP ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            the Knuth-Morris-Pratt algorithm (or KMP algorithm) is a
            string-searching algorithm that searches for occurrences of a "word"
            W within a main "text string" S by employing the observation that
            when a mismatch occurs, the word itself embodies sufficient
            information to determine where the next match could begin, thus
            bypassing re-examination of previously matched characters.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {KMP_code}
          </Code>
        </Box>
        <Divider />
        {/* ETF */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Euler Totient Function
            </Heading>
            <Button size="sm" onClick={onCopy_ETF}>
              {hasCopied_ETF ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            In number theory, Euler's totient function counts the positive
            integers up to a given integer n that are relatively prime to n. It
            is written using the Greek letter phi as φ ( n ) , and may also be
            called Euler's phi function. In other words, it is the number of
            integers k in the range 1 ≤ k ≤ n for which the greatest common
            divisor gcd(n, k) is equal to 1. The integers k of this form are
            sometimes referred to as totatives of n.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {ETF_code}
          </Code>
        </Box>
        <Divider />
        {/* Sieve of Eratostenes */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Sieve of Eratostenes
            </Heading>
            <Button size="sm" onClick={onCopy_SOE}>
              {hasCopied_SOE ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>
            In mathematics, the sieve of Eratosthenes is an ancient algorithm
            for finding all prime numbers up to any given limit.
          </Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {SOE_code}
          </Code>
        </Box>

        <Divider />
        {/* HLD Code */}
        <Box width="100%">
          <HStack justifyContent="space-between" width="100%">
            <Heading as="h2" size="lg" mb={2}>
              Heavy Light Decomposition
            </Heading>
            <Button size="sm" onClick={onCopy_hld_code}>
              {hasCopied_hld_code ? "Copied!" : "Copy"}
            </Button>
          </HStack>
          <Text mb={2}>Operations on trees !</Text>
          <Code
            p={4}
            rounded="md"
            bg="gray.800"
            color="green.300"
            display="block"
            whiteSpace="pre-wrap"
          >
            {hld_code}
          </Code>
        </Box>
        <Divider />
      </VStack>
    </Box>
  );
};

export default SnippetPage;

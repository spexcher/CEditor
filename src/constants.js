// export const LANGUAGE_VERSIONS = {
//   javascript: "18.15.0",
//   typescript: "5.0.3",
//   python: "3.10.0",
//   java: "15.0.2",
//   csharp: "6.12.0",
//   php: "8.2.3",
// };

const fetchLanguageVersions = async () => {
  try {
    // Fetch all runtimes from the Piston API
    const response = await fetch("https://emkc.org/api/v2/piston/runtimes");
    const runtimes = await response.json();

    // Create a mapping of desired languages to their versions
    const languageVersions = {};

    // List of languages we are interested in
    const targetLanguages = [
      "cpp",
      "javascript",
      "typescript",
      "python",
      "java",
      "csharp",
      "php",
    ];

    // runtimes.forEach((runtime) => {
    //   console.log(runtime.aliases);
    //   targetLanguages.forEach((language) => {
    //     runtime.aliases.forEach((alias) => {
    //       if(language==alias) languageVersions[language] = runtime.version;
    //       break;
    //     });
    //     if (language in runtime.aliases || language == runtime.language) {
    //       console.log(runtime.language);
    //       console.log(runtime.version);
    //       console.log(runtime.aliases);
    //       languageVersions[language] = runtime.version;
    //     }
    //   });
    // if (targetLanguages.includes(runtime.language) || targetLanguages in runtime.aliases) {
    //   console.log(runtime.language)
    //   console.log(runtime.version)
    //   console.log( runtime.aliases)
    //   languageVersions[runtime.language] = runtime.version;
    // }
    const targetLanguageSet = new Set(targetLanguages);
    runtimes.forEach((runtime) => {
      const { language, version, aliases } = runtime; // Destructure properties for better readability

      // Check if the runtime's language or any of its aliases are in targetLanguages
      if (targetLanguageSet.has(language)) {
        console.log(language);
        console.log(version);
        console.log(aliases);
        languageVersions[language] = version;
      } else if (aliases.some((alias) => targetLanguageSet.has(alias))) {
        console.log(language);
        console.log(version);
        console.log(aliases);
        languageVersions[language] = version;
      }
    });

    languageVersions["cpp"] = languageVersions["c++"];

    // Return the final language version object
    return languageVersions;
  } catch (error) {
    console.error("Error fetching language versions:", error);
    return {};
  }
};

// export const LANGUAGE_VERSIONS = {
//   'c++':'10.2.0',
//   javascript: "18.15.0",
//   typescript: "5.0.3",
//   python: "3.10.0",
//   java: "15.0.2",
//   csharp: "6.12.0",
//   php: "8.2.3",

// };
// LANGUAGE_VERSIONS = fetchLanguageVersions();

// export const LANGUAGE_VERSIONS;

// Call the function and log the result
const initLanguageVersions = async () => {
  const languageVersions = await fetchLanguageVersions();
  console.log("Fetched Language Versions:", languageVersions);

  // Export the fetched versions
  return languageVersions;
};

// Initialize and export the LANGUAGE_VERSIONS
export const LANGUAGE_VERSIONS = await initLanguageVersions();

export const CODE_SNIPPETS = {
  cpp: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------
#include <bits/stdc++.h>
using namespace std;
void solve();
signed main()
{
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL);                    
    cout.tie(NULL);  
    // Your "pre-logic" here                
    int t = 1;
    // cin >> t;
    for(int i=1;i<=t;i++)
        solve();
    return 0;
}

void solve()
{
    //Start your magic here
}
  
  `,
  javascript: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------
    function greet(name) {
      console.log("Hello, " + name + "!");
    }
    
    greet("Spexcher");
  `,
  typescript: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------
    type Params = {
      name: string;
    };
    
    function greet(data: Params) {
      console.log("Hello, " + data.name + "!");
    }
    
    greet({ name: "Spexcher" });
  `,
  python: `// ----------------------- Competitive Editor ---------------------------------

 #
 #  ❤️ If you enjoy this project, please give it a star on GitHub!
 #  ⭐ https://github.com/spexcher/CEditor
 #  ↗️ Feel free to share it with others who might find it useful!
 # 
 #  Your support helps me improve and create more amazing projects!
 #  Let's build something great together!
 #

 // -------------------- Happy Coding! ---------------------------------

import sys
import os
from collections import defaultdict
from math import inf, sqrt, ceil, floor, pow

# Fast input function
input = sys.stdin.read
data = input().splitlines()

def solve(data):
    # Your logic here

    

if __name__ == "__main__":
    t = 1 
    # t = int(data[0])
    for i in range(t):
        solve(data)`,
  java: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class Main {
    static final long INF = Long.MAX_VALUE;
    static final double PI = 3.1415926535897932384626;

    public static void main(String[] args) throws IOException {
        FastReader reader = new FastReader();
        PrintWriter writer = new PrintWriter(System.out);
        solve(reader, writer);
        writer.close();
    }

    public static void solve(FastReader reader, PrintWriter writer) {
        // Your "pre-logic" here
        int t = 1; // Change as needed
        // t = reader.nextInt();
        for (int i = 0; i < t; i++) {
            //Start your magic here
            
        }
    }



    // Fast IO Below
    // Example Use
    // int n = reader.nextInt();
    // writer.println(n);

    static class FastReader {
        BufferedReader br;
        StringTokenizer st;

        public FastReader() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        String next() {
            while (st == null || !st.hasMoreElements()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }

        int nextInt() {
            return Integer.parseInt(next());
        }

        long nextLong() {
            return Long.parseLong(next());
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }
    }
}
  `,
  csharp: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------
    using System;
    
    namespace HelloWorld
    {
      class Hello { 
        static void Main(string[] args) {
          Console.WriteLine("Hello World in C# from Spexcher");
        }
      }
    }
  `,
  php: `// ----------------------- Competitive Editor ---------------------------------

/*
 *  ❤️ If you enjoy this project, please give it a star on GitHub!
 *  ⭐ https://github.com/spexcher/CEditor
 *  ↗️ Feel free to share it with others who might find it useful!
 * 
 *  Your support helps me improve and create more amazing projects!
 *  Let's build something great together!
 */

 // -------------------- Happy Coding! ---------------------------------


  <?php\n\n$name = 'Spexcher';\necho $name;\n`,
};

// // Now, you can access these snippets in your client-side code
// console.log(CODE_SNIPPETS.javascript);
// console.log(CODE_SNIPPETS.python);

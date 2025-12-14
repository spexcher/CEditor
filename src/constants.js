export const LANGUAGE_VERSIONS = {
  javascript: "18.15.0",
  typescript: "5.0.3",
  python: "3.10.0",
  java: "15.0.2",
  csharp: "6.12.0",
  php: "8.2.3",
  cpp: "10.2.0",
};

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
  python: `# ----------------------- Competitive Editor ---------------------------------

 #
 #  ❤️ If you enjoy this project, please give it a star on GitHub!
 #  ⭐ https://github.com/spexcher/CEditor
 #  ↗️ Feel free to share it with others who might find it useful!
 # 
 #  Your support helps me improve and create more amazing projects!
 #  Let's build something great together!
 #

 # -------------------- Happy Coding! ---------------------------------

import sys
import os
from collections import defaultdict
from math import inf, sqrt, ceil, floor, pow

# Fast input function to handle large inputs
input = sys.stdin.read
data = input().splitlines()

def solve():
    # Implement your solution logic here
    pass

if __name__ == "__main__":
    t = 1  # Default number of test cases (change as per input structure)
    
    if data:  # Ensure data isn't empty
        t = int(data[0])  # First line contains number of test cases

    # Process each test case
    idx = 1  # Start from the second line in data (first line is the number of test cases)
    for i in range(t):
        solve()  # Call the solve function for each test case
`,
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
  php: `
<?php
$name = 'Spexcher';
echo $name;
?>

  `,
};

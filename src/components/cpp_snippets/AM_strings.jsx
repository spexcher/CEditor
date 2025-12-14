const add_multiply_strings_code = `#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
// Function to convert a string of digits into a vector of integers
vector<int> stringToVector(const string &s)
{
  vector<int> digits;
  for (char c : s)
  {
    digits.push_back(c - '0'); // Convert char to int
  }
  return digits;
}

// Function to convert a vector of integers into a string of digits
string vectorToString(const vector<int> &digits)
{
  string s;
  for (int digit : digits)
  {
    s.push_back(digit + '0'); // Convert int to char
  }
  return s;
}

// Function to add two vectors of digits
vector<int> add(const vector<int> &a, const vector<int> &b)
{
  vector<int> result;
  int carry = 0;
  int n = max(a.size(), b.size());
  for (int i = 0; i < n || carry; ++i)
  {
    if (i < a.size())
      carry += a[i];
    if (i < b.size())
      carry += b[i];
    result.push_back(carry % 10);
    carry /= 10;
  }
  return result;
}

// Function to multiply two vectors of digits using elementary multiplication
vector<int> multiply(const vector<int> &a, const vector<int> &b)
{
  vector<int> result(a.size() + b.size(), 0);
  for (int i = 0; i < a.size(); ++i)
  {
    int carry = 0;
    for (int j = 0; j < b.size() || carry; ++j)
    {
      long long current = result[i + j] + carry;
      if (j < b.size())
        current += (long long)a[i] * b[j];
      result[i + j] = current % 10;
      carry = current / 10;
    }
  }
  while (result.size() > 1 && result.back() == 0)
  {
    result.pop_back();
  }
  return result;
}

// Wrapper function to multiply two long integer strings
string multiplyStrings(string s1, string s2)
{
  reverse(all(s1));
  reverse(all(s2));
  vector<int> a = stringToVector(s1);
  vector<int> b = stringToVector(s2);
  if (a.size() == 1 && a[0] == 0 || b.size() == 1 && b[0] == 0)
    return "0";
  vector<int> result = multiply(a, b);
  reverse(result.begin(), result.end());
  return vectorToString(result);
}
string addStrings(string s1, string s2)
{
  reverse(all(s1));
  reverse(all(s2));
  vector<int> a = stringToVector(s1);
  vector<int> b = stringToVector(s2);
  if (a.size() == 1 && a[0] == 0 || b.size() == 1 && b[0] == 0)
    return "0";
  vector<int> result = add(a, b);
  reverse(result.begin(), result.end());
  return vectorToString(result);
}

// string s1 = "99";
// string s2 = "99";

// string result = addStrings(s1, s2);
// string result2 = multiplyStrings(s1, s2);

// cout << "Result of multiplying strings: " << result2 << endl;
// cout << "Result of adding strings: " << result << endl;
`;
export default add_multiply_strings_code;
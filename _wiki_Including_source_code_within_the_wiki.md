# Including Source Code Within the Wiki

Other languages: English, français

To include source code within the wiki, we use the `SyntaxHighlight_GeSHi` extension.  You can easily include a code snippet using the tag `<syntaxhighlight> </syntaxhighlight>`.

## Options of the `<syntaxhighlight>` tag

For a complete list of options, please refer to [this page](link_to_options_page_here).

### `lang` option

The `lang` option defines the language used for syntax highlighting. The default language, if this option is omitted, is C++. The complete list of supported languages is available [here](link_to_languages_page_here).

### `line` option

The `line` option displays line numbers.

## Example

Here is an example of a C++ code snippet created with the `<syntaxhighlight lang="cpp" line> ... </syntaxhighlight>` tag:

```cpp
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sstream>

using namespace std;

void flushIfBig(ofstream &out, ostringstream &oss, int size, bool force = false) {
  if (oss.tellp() >= size) {
    out << oss.str();
    oss.str(""); //reset buffer
  }
}

int main() {
  int buff_size = 50 * 1024 * 1024;
  ofstream out("file.dat");
  ostringstream oss(ostringstream::app);
  oss.precision(5);
  for (int i = 0; i < 100 * buff_size; i++) {
    oss << i << endl;
    flushIfBig(out, oss, buff_size);
  }
  flushIfBig(out, oss, buff_size, true);
  out.close();
}
```

**(Note:  Please replace `link_to_options_page_here` and `link_to_languages_page_here` with the actual links.)**

# Including Source Code Within the Wiki

To include source code in the wiki, we use the SyntaxHighlight_GeSHi extension. You can easily include a source code snippet using the `<syntaxhighlight> </syntaxhighlight>` tag.

## `<syntaxhighlight>` Tag Options

For a list of options, please refer to [this page](link_to_options_page_needed).

### `lang` Option

The `lang` option allows you to define the language used for syntax highlighting. The default language, if this parameter is omitted, is C++. The list of supported languages is available [here](link_to_supported_languages_needed).

### `line` Option

The `line` option allows you to display line numbers.

## Example

Here is an example of C++ code created with the `<syntaxhighlight lang="cpp" line> ... </syntaxhighlight>` tag:

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

**(Note:  Please replace `link_to_options_page_needed` and `link_to_supported_languages_needed` with the actual links.)**

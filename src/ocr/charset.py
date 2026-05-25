from string import ascii_lowercase, ascii_uppercase, digits

BLANK_TOKEN = "<BLANK>"
FULL_CHARACTERS = " " + digits + ascii_uppercase + ascii_lowercase
LOWERCASE_CHARACTERS = ascii_lowercase


class Charset:
    def __init__(self, characters=FULL_CHARACTERS, blank_token=BLANK_TOKEN, name="custom"):
        self.characters = characters
        self.blank_token = blank_token
        self.name = name
        self.blank_index = 0
        self.index_to_char = {index + 1: character for index, character in enumerate(characters)}
        self.char_to_index = {character: index for index, character in self.index_to_char.items()}

    @property
    def size(self):
        return len(self.characters) + 1

    def contains(self, text):
        return all(character in self.char_to_index for character in text)

    def encode(self, text):
        if not self.contains(text):
            invalid = sorted({character for character in text if character not in self.char_to_index})
            raise ValueError(f"Unsupported characters in text: {invalid}")
        return [self.char_to_index[character] for character in text]

    def decode(self, indices):
        characters = []
        for index in indices:
            if index == self.blank_index:
                continue
            if index not in self.index_to_char:
                raise ValueError(f"Unknown character index: {index}")
            characters.append(self.index_to_char[index])
        return "".join(characters)

    def to_metadata(self):
        return {
            "name": self.name,
            "blank_token": self.blank_token,
            "blank_index": self.blank_index,
            "characters": self.characters,
        }


FULL_CHARSET = Charset(characters=FULL_CHARACTERS, name="full")
LOWERCASE_CHARSET = Charset(characters=LOWERCASE_CHARACTERS, name="lowercase")
DEFAULT_CHARSET = FULL_CHARSET
CHARSET_REGISTRY = {
    FULL_CHARSET.name: FULL_CHARSET,
    LOWERCASE_CHARSET.name: LOWERCASE_CHARSET,
}


def get_charset(name):
    if name not in CHARSET_REGISTRY:
        raise ValueError(f"Unknown charset: {name}")
    return CHARSET_REGISTRY[name]

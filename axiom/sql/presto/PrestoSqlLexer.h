/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Generated from PrestoSql.g4 by ANTLR 4.9.3

#pragma once

#include "antlr4-runtime.h"

namespace facebook::velox::sql {

class PrestoSqlLexer : public antlr4::Lexer {
 public:
  enum {
    T__0 = 1,
    T__1 = 2,
    T__2 = 3,
    T__3 = 4,
    T__4 = 5,
    T__5 = 6,
    T__6 = 7,
    T__7 = 8,
    T__8 = 9,
    ENGLISH_TOKEN = 10,
    ADD = 11,
    ADMIN = 12,
    ALL = 13,
    ALTER = 14,
    ANALYZE = 15,
    AND = 16,
    ANY = 17,
    ARRAY = 18,
    AS = 19,
    ASC = 20,
    AT = 21,
    BEFORE = 22,
    BERNOULLI = 23,
    BETWEEN = 24,
    BY = 25,
    CALL = 26,
    CALLED = 27,
    CASCADE = 28,
    CASE = 29,
    CAST = 30,
    CATALOGS = 31,
    COLUMN = 32,
    COLUMNS = 33,
    COMMENT = 34,
    COMMIT = 35,
    COMMITTED = 36,
    CONSTRAINT = 37,
    CREATE = 38,
    CROSS = 39,
    CUBE = 40,
    CURRENT = 41,
    CURRENT_DATE = 42,
    CURRENT_ROLE = 43,
    CURRENT_TIME = 44,
    CURRENT_TIMESTAMP = 45,
    CURRENT_USER = 46,
    DATA = 47,
    DATE = 48,
    DAY = 49,
    DEALLOCATE = 50,
    DEFINER = 51,
    DELETE = 52,
    DESC = 53,
    DESCRIBE = 54,
    DETERMINISTIC = 55,
    DISABLED = 56,
    DISTINCT = 57,
    DISTRIBUTED = 58,
    DROP = 59,
    ELSE = 60,
    ENABLED = 61,
    END = 62,
    ENFORCED = 63,
    ESCAPE = 64,
    EXCEPT = 65,
    EXCLUDING = 66,
    EXECUTE = 67,
    EXISTS = 68,
    EXPLAIN = 69,
    EXTRACT = 70,
    EXTERNAL = 71,
    FALSE = 72,
    FETCH = 73,
    FILTER = 74,
    FIRST = 75,
    FOLLOWING = 76,
    FOR = 77,
    FORMAT = 78,
    FROM = 79,
    FULL = 80,
    FUNCTION = 81,
    FUNCTIONS = 82,
    GRANT = 83,
    GRANTED = 84,
    GRANTS = 85,
    GRAPHVIZ = 86,
    GROUP = 87,
    GROUPING = 88,
    GROUPS = 89,
    HAVING = 90,
    HOUR = 91,
    IF = 92,
    IGNORE = 93,
    IN = 94,
    INCLUDING = 95,
    INNER = 96,
    INPUT = 97,
    INSERT = 98,
    INTERSECT = 99,
    INTERVAL = 100,
    INTO = 101,
    INVOKER = 102,
    IO = 103,
    IS = 104,
    ISOLATION = 105,
    JSON = 106,
    JOIN = 107,
    KEY = 108,
    LANGUAGE = 109,
    LAST = 110,
    LATERAL = 111,
    LEFT = 112,
    LEVEL = 113,
    LIKE = 114,
    LIMIT = 115,
    LOCALTIME = 116,
    LOCALTIMESTAMP = 117,
    LOGICAL = 118,
    MAP = 119,
    MATERIALIZED = 120,
    MINUTE = 121,
    MONTH = 122,
    NAME = 123,
    NATURAL = 124,
    NFC = 125,
    NFD = 126,
    NFKC = 127,
    NFKD = 128,
    NO = 129,
    NONE = 130,
    NORMALIZE = 131,
    NOT = 132,
    NULL_LITERAL = 133,
    NULLIF = 134,
    NULLS = 135,
    OF = 136,
    OFFSET = 137,
    ON = 138,
    ONLY = 139,
    OPTION = 140,
    OR = 141,
    ORDER = 142,
    ORDINALITY = 143,
    OUTER = 144,
    OUTPUT = 145,
    OVER = 146,
    PARTITION = 147,
    PARTITIONS = 148,
    POSITION = 149,
    PRECEDING = 150,
    PREPARE = 151,
    PRIMARY = 152,
    PRIVILEGES = 153,
    PROPERTIES = 154,
    RANGE = 155,
    READ = 156,
    RECURSIVE = 157,
    REFRESH = 158,
    RELY = 159,
    RENAME = 160,
    REPEATABLE = 161,
    REPLACE = 162,
    RESET = 163,
    RESPECT = 164,
    RESTRICT = 165,
    RETURN = 166,
    RETURNS = 167,
    REVOKE = 168,
    RIGHT = 169,
    ROLE = 170,
    ROLES = 171,
    ROLLBACK = 172,
    ROLLUP = 173,
    ROW = 174,
    ROWS = 175,
    SCHEMA = 176,
    SCHEMAS = 177,
    SECOND = 178,
    SECURITY = 179,
    SELECT = 180,
    SERIALIZABLE = 181,
    SESSION = 182,
    SET = 183,
    SETS = 184,
    SHOW = 185,
    SOME = 186,
    SQL = 187,
    START = 188,
    STATS = 189,
    SUBSTRING = 190,
    SYSTEM = 191,
    SYSTEM_TIME = 192,
    SYSTEM_VERSION = 193,
    TABLE = 194,
    TABLES = 195,
    TABLESAMPLE = 196,
    TEMPORARY = 197,
    TEXT = 198,
    THEN = 199,
    TIME = 200,
    TIMESTAMP = 201,
    TO = 202,
    TRANSACTION = 203,
    TRUE = 204,
    TRUNCATE = 205,
    TRY_CAST = 206,
    TYPE = 207,
    UESCAPE = 208,
    UNBOUNDED = 209,
    UNCOMMITTED = 210,
    UNION = 211,
    UNIQUE = 212,
    UNNEST = 213,
    UPDATE = 214,
    USE = 215,
    USER = 216,
    USING = 217,
    VALIDATE = 218,
    VALUES = 219,
    VERBOSE = 220,
    VERSION = 221,
    VIEW = 222,
    WHEN = 223,
    WHERE = 224,
    WITH = 225,
    WORK = 226,
    WRITE = 227,
    YEAR = 228,
    ZONE = 229,
    EQ = 230,
    NEQ = 231,
    LT = 232,
    LTE = 233,
    GT = 234,
    GTE = 235,
    PLUS = 236,
    MINUS = 237,
    ASTERISK = 238,
    SLASH = 239,
    PERCENT = 240,
    CONCAT = 241,
    STRING = 242,
    UNICODE_STRING = 243,
    BINARY_LITERAL = 244,
    INTEGER_VALUE = 245,
    DECIMAL_VALUE = 246,
    DOUBLE_VALUE = 247,
    IDENTIFIER = 248,
    DIGIT_IDENTIFIER = 249,
    QUOTED_IDENTIFIER = 250,
    BACKQUOTED_IDENTIFIER = 251,
    TIME_WITH_TIME_ZONE = 252,
    TIMESTAMP_WITH_TIME_ZONE = 253,
    DOUBLE_PRECISION = 254,
    SIMPLE_COMMENT = 255,
    BRACKETED_COMMENT = 256,
    WS = 257,
    UNRECOGNIZED = 258
  };

  explicit PrestoSqlLexer(antlr4::CharStream* input);
  ~PrestoSqlLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames()
      const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

 private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

} // namespace facebook::velox::sql

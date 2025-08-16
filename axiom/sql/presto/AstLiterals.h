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
#pragma once

#include "AstNode.h"
#include "AstVisitor.h"
#include <string>
#include <vector>

namespace facebook::velox::sql {

class Literal : public Expression {
public:
  explicit Literal(NodeType type) : Expression(type) {}
};

using LiteralPtr = std::shared_ptr<Literal>;

class BooleanLiteral : public Literal {
public:
  explicit BooleanLiteral(bool value)
      : Literal(NodeType::kBooleanLiteral), value_(value) {}
  
  bool getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  bool value_;
};

class StringLiteral : public Literal {
public:
  explicit StringLiteral(const std::string& value)
      : Literal(NodeType::kStringLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class BinaryLiteral : public Literal {
public:
  explicit BinaryLiteral(const std::string& value)
      : Literal(NodeType::kBinaryLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class CharLiteral : public Literal {
public:
  explicit CharLiteral(const std::string& value)
      : Literal(NodeType::kCharLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class LongLiteral : public Literal {
public:
  explicit LongLiteral(int64_t value)
      : Literal(NodeType::kLongLiteral), value_(value) {}
  
  int64_t getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  int64_t value_;
};

class DoubleLiteral : public Literal {
public:
  explicit DoubleLiteral(double value)
      : Literal(NodeType::kDoubleLiteral), value_(value) {}
  
  double getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  double value_;
};

class DecimalLiteral : public Literal {
public:
  explicit DecimalLiteral(const std::string& value)
      : Literal(NodeType::kDecimalLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class GenericLiteral : public Literal {
public:
  GenericLiteral(const std::string& type, const std::string& value)
      : Literal(NodeType::kGenericLiteral), type_(type), value_(value) {}
  
  const std::string& getType() const { return type_; }
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string type_;
  std::string value_;
};

class NullLiteral : public Literal {
public:
  NullLiteral() : Literal(NodeType::kNullLiteral) {}
  void accept(AstVisitor* visitor) override;
};

class TimeLiteral : public Literal {
public:
  explicit TimeLiteral(const std::string& value)
      : Literal(NodeType::kTimeLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class TimestampLiteral : public Literal {
public:
  explicit TimestampLiteral(const std::string& value)
      : Literal(NodeType::kTimestampLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

class IntervalLiteral : public Literal {
public:
  enum class Sign { POSITIVE, NEGATIVE };
  enum class IntervalField { 
    YEAR, MONTH, DAY, HOUR, MINUTE, SECOND 
  };
  
  IntervalLiteral(
      const std::string& value,
      Sign sign,
      IntervalField startField,
      std::optional<IntervalField> endField = std::nullopt)
      : Literal(NodeType::kIntervalLiteral),
        value_(value),
        sign_(sign),
        startField_(startField),
        endField_(endField) {}
  
  const std::string& getValue() const { return value_; }
  Sign getSign() const { return sign_; }
  IntervalField getStartField() const { return startField_; }
  const std::optional<IntervalField>& getEndField() const { return endField_; }
  
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
  Sign sign_;
  IntervalField startField_;
  std::optional<IntervalField> endField_;
};

class EnumLiteral : public Literal {
public:
  explicit EnumLiteral(const std::string& value)
      : Literal(NodeType::kEnumLiteral), value_(value) {}
  
  const std::string& getValue() const { return value_; }
  void accept(AstVisitor* visitor) override;

private:
  std::string value_;
};

} // namespace facebook::velox::sql
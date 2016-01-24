// Copyright 2015 Byungkuk Choi

#ifndef MLCORE_EXCEPTIONS_H_
#define MLCORE_EXCEPTIONS_H_

#include <exception>
#include <string>

namespace ML {

class BadInputException : public std::exception {
 public:
  explicit BadInputException(std::string const& msg)
      : std::exception(), _msg(msg) {}

  virtual const char* what() const throw() { return _msg.c_str(); }

 private:
  std::string _msg;
};

}  // namespace ML

#endif  // MLCORE_EXCEPTIONS_H_

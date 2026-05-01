#!/usr/bin/env node

/**
 * declarative-parallel-dsl CLI
 *
 * Entry point for the DSL runner.
 * Drop your DSL source files and implementation modules into src/.
 */

const [,, ...args] = process.argv;

if (args.length === 0) {
  console.log("Usage: dsl <file.dsl>");
  console.log("");
  console.log("declarative-parallel-dsl — a minimal CLI for running declarative parallel programs.");
  process.exit(0);
}

console.log(`Running DSL file: ${args[0]}`);
console.log("(Implementation pending — drop your DSL files here!)");

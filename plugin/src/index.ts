// SPDX-License-Identifier: AGPL-3.0-or-later
import { PluginContext } from "@lmstudio/sdk";
import { configSchematics, globalConfigSchematics } from "./config";
import { generate } from "./generator";

export async function main(context: PluginContext) {
  context.withConfigSchematics(configSchematics)
    .withGlobalConfigSchematics(globalConfigSchematics)
    .withGenerator(generate);
}

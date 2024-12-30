#version 450
/* Copyright (c) 2024, Huawei Technologies Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
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

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 3) in vec3 in_trans;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_nrml;

layout(binding = 0) uniform UboSene
{
    mat4 projection;
}uboScene;

void main()
{
    gl_Position = uboScene.projection * vec4(in_position+in_trans, 1.0);

    out_color = in_color;
    out_nrml = normalize(in_normal);
}
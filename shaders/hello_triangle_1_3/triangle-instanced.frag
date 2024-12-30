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

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_nrml;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 lightDir = normalize(vec3(-1,-1,-1));
	float diffuse = max(dot(in_nrml,lightDir),0);
	out_color = vec4(in_color*diffuse, 1.0);
}
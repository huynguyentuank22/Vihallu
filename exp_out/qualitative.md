# Qualitative Counterfactual Examples

Model: **Qwen_Qwen3_1.7B**

Selected 20 examples with largest probability shifts.

---
## Example 1  |  Bucket C (no→extrinsic)

**ID:** `3278102c-266e-4ace-b1d7-a415b74324ef`

**Context (truncated):** Ngày 31 tháng 3 năm 2014, Epic Records thông báo rằng album di sản tiếp theo của Jackson, Xscape, bao gồm 8 bài hát hoàn toàn mới được chọn lọc từ các bản ghi âm nháp mà ông thực hiện lúc còn sống,...

**Prompt:** Epc Recoords đã cho ra mắt tuyển tập tên giì của Jacksn đê lại?

**Original Response:** Epic Records đã cho ra mắt tuyển tập tên là Xscape của Jackson.

**Counterfactual Response:** Epic Records đã cho ra mắt tuyển tập tên là Xscape của Jackson. Ngoài ra, khu vực sản xuất album này nằm gần biển.

**Diff:** Epic Records đã cho ra mắt tuy … tập tên là Xscape của Jackson. **+Ngoài ra, khu vực sản xuất album này nằm gần biển.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9963 | 0.0128 |
| P(intrinsic) | 0.0028 | 0.0386 |
| P(extrinsic) | 0.0009 | 0.9487 |
| Token edits | 11 | — |
| Prob shift Σ|Δp| | 1.9670 | — |

**Rationale:** Added a new sentence not present in context, geographically referencing the production area, which is not mentioned or implied in the context.

---
## Example 2  |  Bucket C (no→extrinsic)

**ID:** `b64694ba-c015-4d36-973b-8afae22d2324`

**Context (truncated):** Hoa Kỳ và Liên Xô nổi lên thành hai siêu cường đối nghịch nhau sau Chiến tranh thế giới thứ hai và khởi động một cuộc Chiến tranh Lạnh, đối đầu nhau gián tiếp trong cuộc chạy đua vũ trang và chạy đ...

**Prompt:** Kể từ khi Chiến tranh thế giới thứ hai kết thúc, Hoa Kỳ đã trải qua thời kỳ suy thoái kinh tế tệ hại nhất là vào khoảng thời gian nào?

**Original Response:** Hoa Kỳ trải qua thời kỳ suy thoái kinh tế tệ hại nhất kể từ Chiến tranh thế giới thứ hai vào cuối thập niên 2000.

**Counterfactual Response:** Hoa Kỳ trải qua thời kỳ suy thoái kinh tế tệ hại nhất kể từ Chiến tranh thế giới thứ hai vào cuối thập niên 2000. Diện tích khoảng 500 km².

**Diff:** Hoa Kỳ trải qua thời kỳ suy th … ứ hai vào cuối thập niên 2000. **+Diện tích khoảng 500 km².**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | intrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9907 | 0.0122 |
| P(intrinsic) | 0.0086 | 0.7490 |
| P(extrinsic) | 0.0006 | 0.2389 |
| Token edits | 5 | — |
| Prob shift Σ|Δp| | 1.9572 | — |

**Rationale:** Added a sentence not present in the context and not logically derived from it, while maintaining the original response and avoiding prohibited connectors.

---
## Example 3  |  Bucket B (extrinsic→no)

**ID:** `005197c9-45f9-44db-8442-5398d2f21999`

**Context (truncated):** Động thực vật tại Nouvelle-Calédonie bắt nguồn từ các loài tổ tiên bị cô lập tại khu vực khi quần đảo tách khỏi Gondwana hàng chục triệu năm trước. Nouvelle-Calédonie có nhiều loài thực vật hạt trầ...

**Prompt:** Đã có bao nhiêu loài Araucaria được tìm thấy tại Nouvelle-Calédonie?

**Original Response:** Có 35 loài Araucaria được tìm thấy tại Nouvelle-Calédonie. Nơi này cũng nổi tiếng với hệ sinh thái rạn san hô lớn thứ hai thế giới, chỉ sau rạn san hô Great Barrier của Úc.

**Counterfactual Response:** Có 35 loài Araucaria được tìm thấy tại Nouvelle-Calédonie.

**Diff:** Có 35 loài Araucaria được tìm thấy tại Nouvelle-Calédonie. ~~Nơi này cũng nổi tiếng với hệ sinh thái rạn san hô lớn thứ hai thế giới, chỉ sau rạn san hô Great Barrier của Úc.~~

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | extrinsic | no |
| Expected | extrinsic | no |
| P(no) | 0.0144 | 0.9927 |
| P(intrinsic) | 0.0695 | 0.0064 |
| P(extrinsic) | 0.9161 | 0.0009 |
| Token edits | 26 | — |
| Prob shift Σ|Δp| | 1.9566 | — |

---
## Example 4  |  Bucket A (intrinsic→no)

**ID:** `fceac3c1-e8a4-485f-8276-443919ed8260`

**Context (truncated):** Chi Dissostichus có hai loài, Dissostichus mawsoni và Dissostichus eleginoides. Cả hai sống ở dưới đáy có độ sâu dao động từ 100–3.000 m (328–9.843 ft) và có thể sinh trưởng tới chiều dài 2 m (7 ft...

**Prompt:** Chii Dissotisuchus gồm hai loài nàoo?

**Original Response:** Chi Dissostichus thực ra chỉ bao gồm một loài duy nhất là Dissostichus eleginoides, sống cả ở Nam Cực và cận Nam Cực, không có loài nào tên là Dissostichus mawsoni.

**Counterfactual Response:** Chi Dissostichus gồm hai loài: Dissostichus mawsoni và Dissostichus eleginoides.

**Diff:** Chi Dissostichus ~~thực ra chỉ bao~~ gồm ~~một loài duy nhất là~~ → **hai loài:** Dissostichus ~~eleginoides, sống cả ở Nam Cực~~ → **mawsoni** và ~~cận Nam Cực, không có loài nào tên là~~ Dissostichus ~~mawsoni.~~ → **eleginoides.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | intrinsic | no |
| Expected | intrinsic | no |
| P(no) | 0.0206 | 0.9932 |
| P(intrinsic) | 0.9414 | 0.0049 |
| P(extrinsic) | 0.0380 | 0.0022 |
| Token edits | 25 | — |
| Prob shift Σ|Δp| | 1.9448 | — |

**Rationale:** Corrects intrinsic contradiction by aligning with context that mentions two species.

---
## Example 5  |  Bucket B (extrinsic→no)

**ID:** `9df37c43-478e-4305-bd6c-8bfd692e9359`

**Context (truncated):** Canada là quốc gia đa tôn giáo, bao gồm nhiều tín ngưỡng và phong tục. Theo điều tra năm 2011, 67,3% người Canada nhận mình là tín hữu Kitô giáo; trong đó Công giáo Rôma là giáo hội lớn nhất với 38...

**Prompt:** Mặc dù Hồi giáo là tôn giáo lớn nhất ở Canada, số dân chúng tham gia giáo Rôma chiếm bao nhiêu phần trăm dân số?

**Original Response:** Công giáo Rôma chiếm 38,7% dân số Canada, trong khi người Hồi giáo chiếm 3,2%. Thú vị là Canada là một trong những quốc gia có tỷ lệ chấp nhận người nhập cư cao nhất thế giới, với chính sách đa văn hóa mạnh mẽ.

**Counterfactual Response:** Công giáo Rôma chiếm 38,7% dân số Canada, trong khi người Hồi giáo chiếm 3,2%.

**Diff:** Công giáo Rôma chiếm 38,7% dân … khi người Hồi giáo chiếm 3,2%. ~~Thú vị là Canada là một trong những quốc gia có tỷ lệ chấp nhận người nhập cư cao nhất thế giới, với chính sách đa văn hóa mạnh mẽ.~~

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | extrinsic | no |
| Expected | extrinsic | no |
| P(no) | 0.0125 | 0.9814 |
| P(intrinsic) | 0.0947 | 0.0116 |
| P(extrinsic) | 0.8928 | 0.0067 |
| Token edits | 30 | — |
| Prob shift Σ|Δp| | 1.9381 | — |

---
## Example 6  |  Bucket C (no→extrinsic)

**ID:** `ae851d62-70e8-40c6-8d67-644d1ab94377`

**Context (truncated):** Năm 1988, Trump mua lại Taj Mahal Casino sau một thương vụ với Merv Griffin và Resorts International, dẫn tới một khoản nợ lớn nên tới năm 1989, Trump không có khả năng trả các khoản vay. Mặc dù ôn...

**Prompt:** Việc không thể trả được các khoản vay đã khiến Trump phá sản vào năm nào?

**Original Response:** Trump phá sản vào năm 1991 do không thể trả được các khoản vay liên quan đến Taj Mahal Casino.

**Counterfactual Response:** Trump phá sản vào năm 1991 do không thể trả được các khoản vay liên quan đến Taj Mahal Casino. Ngoài ra, dân số khoảng 2 triệu người.

**Diff:** Trump phá sản vào năm 1991 do  … iên quan đến Taj Mahal Casino. **+Ngoài ra, dân số khoảng 2 triệu người.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9933 | 0.0245 |
| P(intrinsic) | 0.0038 | 0.0202 |
| P(extrinsic) | 0.0029 | 0.9556 |
| Token edits | 8 | — |
| Prob shift Σ|Δp| | 1.9379 | — |

**Rationale:** Added a sentence not present in the context and not logically derived from it, while maintaining the original response and following the rule of adding exactly one extrinsic statement at the end.

---
## Example 7  |  Bucket B (extrinsic→no)

**ID:** `aead8688-4ca5-4c2b-a79a-db87b7f19c2b`

**Context (truncated):** Các nam tước địa phương bỏ rơi Isaac, vốn dự định giảng hòa với Richard, hợp quân với ông trong cuộc Thập tự chinh và gả con gái của mình cho một người được lựa chọn bởi Richard. Nhưng Isaac đổi ý ...

**Prompt:** Richard chiếm được toàn bộ hòn đảo vào ngày nào, mặc dù ông đã từ bỏ ý định chiếm đóng và chỉ muốn giảng hòa với Isaac?

**Original Response:** Richard chiếm được toàn bộ hòn đảo vào ngày 1 tháng 6. Điều thú vị là hòn đảo này sau đó trở thành một trong những trung tâm thương mại quan trọng của Địa Trung Hải, với các thương nhân từ Venice và Genoa thường xuyên qua lại.

**Counterfactual Response:** Richard chiếm được toàn bộ hòn đảo vào ngày 1 tháng 6.

**Diff:** Richard chiếm được toàn bộ hòn đảo vào ngày 1 tháng 6. ~~Điều thú vị là hòn đảo này sau đó trở thành một trong những trung tâm thương mại quan trọng của Địa Trung Hải, với các thương nhân từ Venice và Genoa thường xuyên qua lại.~~

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | extrinsic | no |
| Expected | extrinsic | no |
| P(no) | 0.0080 | 0.9766 |
| P(intrinsic) | 0.0542 | 0.0221 |
| P(extrinsic) | 0.9378 | 0.0011 |
| Token edits | 36 | — |
| Prob shift Σ|Δp| | 1.9373 | — |

---
## Example 8  |  Bucket C (no→extrinsic)

**ID:** `6ee33b82-5e5c-4d1b-b05a-91279ba51d5f`

**Context (truncated):** Theo báo Pravda của Nga cho là Quỹ Dân chủ Mỹ (NED - National Endowment for Democracy) có mặt trên khắp đất nước Nga, thâm nhập vào diễn biến chính trị ở Nga hiện nay, giúp đỡ "trung tâm tin tức qu...

**Prompt:** NED dã tài trợ cho các tổ chức thanh thiếu niên của Nga vơi mục đích nao?

**Original Response:** NED đã tài trợ cho các tổ chức thanh thiếu niên của Nga với mục đích "bồi dưỡng thế hệ lãnh đạo mới cho đất nước Nga".

**Counterfactual Response:** NED đã tài trợ cho các tổ chức thanh thiếu niên của Nga với mục đích "bồi dưỡng thế hệ lãnh đạo mới cho đất nước Nga". Ngoài ra, khu vực này có hệ thống giáo dục phát triển mạnh mẽ.

**Diff:** NED đã tài trợ cho các tổ chức … ãnh đạo mới cho đất nước Nga". **+Ngoài ra, khu vực này có hệ thống giáo dục phát triển mạnh mẽ.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9970 | 0.0295 |
| P(intrinsic) | 0.0014 | 0.0579 |
| P(extrinsic) | 0.0016 | 0.9126 |
| Token edits | 14 | — |
| Prob shift Σ|Δp| | 1.9351 | — |

**Rationale:** The added sentence 'Ngoài ra, khu vực này có hệ thống giáo dục phát triển mạnh mẽ' is not present in the context, does not contradict it, and is added only at the end as required. It introduces new information not derived from or implied by the context.

---
## Example 9  |  Bucket A (intrinsic→no)

**ID:** `c7a64990-1dcf-40bf-b463-94a7be5cc1b9`

**Context (truncated):** Cho đến cuối thế kỷ XVIII, từ Do Thái và theo đạo Do Thái được xem là đồng nghĩa trong thực tế, và đạo Do Thái là yếu tố chính thống nhất người Do Thái mặc dù mức độ theo đạo đó có khác nhau. Trong...

**Prompt:** Từ Do Thái và theo đạo Do Thái được xem là đồng nghĩa trong thực tế vào thời gian nào?

**Original Response:** Vào đầu thế kỷ XIX, từ Do Thái và theo đạo Do Thái vẫn được xem là đồng nghĩa, bất chấp những thay đổi xã hội diễn ra trước đó.

**Counterfactual Response:** Cho đến cuối thế kỷ XVIII, từ Do Thái và theo đạo Do Thái được xem là đồng nghĩa trong thực tế.

**Diff:** ~~Vào đầu~~ → **Cho đến cuối** thế kỷ ~~XIX,~~ → **XVIII,** từ Do Thái và theo đạo Do Thái ~~vẫn~~ được xem là đồng ~~nghĩa, bất chấp những thay đổi xã hội diễn ra trước đó.~~ → **nghĩa trong thực tế.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | intrinsic | no |
| Expected | intrinsic | no |
| P(no) | 0.0310 | 0.9985 |
| P(intrinsic) | 0.9129 | 0.0015 |
| P(extrinsic) | 0.0561 | 0.0000 |
| Token edits | 17 | — |
| Prob shift Σ|Δp| | 1.9350 | — |

**Rationale:** Context chỉ rõ rằng từ Do Thái và theo đạo Do Thái được xem là đồng nghĩa đến cuối thế kỷ XVIII, nên sửa lại thời gian để phù hợp với Context

---
## Example 10  |  Bucket C (no→extrinsic)

**ID:** `1d58941a-7fed-4779-9486-85735059a4be`

**Context (truncated):** Sự phản đối từ các giảo dân mất nhiều thời gian để trở nên công khai. Những chống đối tập trung vào hai điều: quyền triệu tập quân đội của nhà vua, và quyền thu thuế của ông. Tại Nghị viện Salisbur...

**Prompt:** Những sự chống đối của dân chúng với vua Edward chủ yếu ở những khía cạnh nào?

**Original Response:** Những sự chống đối của dân chúng với vua Edward chủ yếu tập trung vào quyền triệu tập quân đội của nhà vua và quyền thu thuế của ông, như được phản ánh trong phản đối của Roger Bigod và Humphrey de Bohun.

**Counterfactual Response:** Những sự chống đối của dân chúng với vua Edward chủ yếu tập trung vào quyền triệu tập quân đội của nhà vua và quyền thu thuế của ông, như được phản ánh trong phản đối của Roger Bigod và Humphrey de Bohun. Ngoài ra, khu vực này nằm gần biển.

**Diff:** Những sự chống đối của dân chú … er Bigod và Humphrey de Bohun. **+Ngoài ra, khu vực này nằm gần biển.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9758 | 0.0145 |
| P(intrinsic) | 0.0091 | 0.0147 |
| P(extrinsic) | 0.0152 | 0.9707 |
| Token edits | 8 | — |
| Prob shift Σ|Δp| | 1.9225 | — |

**Rationale:** Added a sentence not present in the context and not logically derived from it, while maintaining the original response structure and avoiding forbidden connectors.

---
## Example 11  |  Bucket A (intrinsic→no)

**ID:** `2ee147ab-4534-490b-8952-072d613b57d2`

**Context (truncated):** Năm 1914 Erwin Schrödinger đạt được văn bằng Habilitation (venia legendi). Giai đoạn 1914 đến 1918 ông tham gia chiến tranh với vị trí là sĩ quan chính thức trong đội pháo binh phòng thủ của các lâ...

**Prompt:** Erwin Schrödinger tham gia Chiến tranh Thế giới thứ I với chức vụ gì?

**Original Response:** Erwin Schrödinger tham gia Chiến tranh Thế giới thứ I với chức vụ là một nhà khoa học nghiên cứu vũ khí hóa học cho quân đội Áo.

**Counterfactual Response:** Erwin Schrödinger tham gia Chiến tranh Thế giới thứ I với chức vụ là sĩ quan chính thức trong đội pháo binh phòng thủ của các lâu đài thuộc Áo.

**Diff:** Erwin Schrödinger tham gia Chi …  Thế giới thứ I với chức vụ là ~~một nhà khoa học nghiên cứu vũ khí hóa học cho quân~~ → **sĩ quan chính thức trong** đội **+pháo binh phòng thủ của các lâu đài thuộc** Áo.

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | intrinsic | no |
| Expected | intrinsic | no |
| P(no) | 0.0379 | 0.9971 |
| P(intrinsic) | 0.8367 | 0.0012 |
| P(extrinsic) | 0.1253 | 0.0018 |
| Token edits | 15 | — |
| Prob shift Σ|Δp| | 1.9182 | — |

**Rationale:** Context explicitly states Schrödinger held the position of 'sĩ quan chính thức' in the Austrian artillery defense units; the original response introduced false information about being a weapons research scientist.

---
## Example 12  |  Bucket C (no→extrinsic)

**ID:** `3edfc09e-30e4-4241-a0c0-71da977604a6`

**Context (truncated):** Theo một chương trình truyền hình vào năm 2012, người nhập cư Ethiopia là nữ giới có thể đã được cấp thuốc ngừa thai Depo-Provera mà không có giải thích đầy đủ về tác dụng của thuốc, mặc dù Bộ Y tế...

**Prompt:** Người nhập cư Ethiopia nữ giớ được nhận thuốc ngừa thai loại nào?

**Original Response:** Người nhập cư Ethiopia nữ giới đã được nhận thuốc ngừa thai Depo-Provera.

**Counterfactual Response:** Người nhập cư Ethiopia nữ giới đã được nhận thuốc ngừa thai Depo-Provera. Khu vực này nằm gần biển.

**Diff:** Người nhập cư Ethiopia nữ giới …  thuốc ngừa thai Depo-Provera. **+Khu vực này nằm gần biển.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9975 | 0.0404 |
| P(intrinsic) | 0.0017 | 0.1761 |
| P(extrinsic) | 0.0008 | 0.7837 |
| Token edits | 6 | — |
| Prob shift Σ|Δp| | 1.9145 | — |

**Rationale:** Added a geographically extrinsic statement 'Khu vực này nằm gần biển' which is not in the context but does not contradict it, satisfying EXTRINSIC requirement with minimal edit.

---
## Example 13  |  Bucket C (no→extrinsic)

**ID:** `6ae00802-0ee4-46ff-9bcd-ee835994b36a`

**Context (truncated):** Không lâu sau khi Jackson mất, vào ngày 25 tháng 6 năm 2009, MTV quay lại định dạng trình chiếu video âm nhạc để kỷ niệm đến ông. Kênh trình chiếu video âm nhạc của ông hàng giờ liền, cùng nhiều bả...

**Prompt:** Tại sao MTV lại quyết định không phát sóng bất kỳ video âm nhạc nào của Jackson sau khi ông qua đời vào ngày 25 tháng 6 năm 2009?

**Original Response:** MTV quyết định phát sóng video âm nhạc của Jackson hàng giờ liền sau khi ông qua đời vào ngày 25 tháng 6 năm 2009, không phải là không phát sóng.

**Counterfactual Response:** MTV quyết định phát sóng video âm nhạc của Jackson hàng giờ liền sau khi ông qua đời vào ngày 25 tháng 6 năm 2009, không phải là không phát sóng. Khu vực này nằm gần biển.

**Diff:** MTV quyết định phát sóng video … không phải là không phát sóng. **+Khu vực này nằm gần biển.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | intrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9662 | 0.0089 |
| P(intrinsic) | 0.0250 | 0.5840 |
| P(extrinsic) | 0.0088 | 0.4070 |
| Token edits | 6 | — |
| Prob shift Σ|Δp| | 1.9144 | — |

**Rationale:** Added a sentence about the location being near the sea, which is entirely extrinsic and does not contradict the context.

---
## Example 14  |  Bucket B (extrinsic→no)

**ID:** `fc5be590-f133-45e8-a646-536508da1139`

**Context (truncated):** Tổ chức của thư viện được chia thành nhiều phòng ban nhỏ, chia sẻ các chức năng hành chính, bảo quản, hợp tác, nghiên cứu... Năm 2007, tổng số nhân viên của Thư viện Quốc gia Pháp lên tới 2.662 ngư...

**Prompt:** Năm 2007 nhân viên của Thư viện Quốc gia Pháp có tổng cộng bao nhiêu người?

**Original Response:** Năm 2007, Thư viện Quốc gia Pháp có tổng cộng 2.662 nhân viên. Thư viện này nằm ở Paris và là một trong những thư viện lớn nhất thế giới, nổi tiếng với bộ sưu tập sách và tài liệu quý hiếm.

**Counterfactual Response:** Năm 2007, Thư viện Quốc gia Pháp có tổng cộng 2.662 nhân viên.

**Diff:** Năm 2007, Thư viện Quốc gia Ph …  có tổng cộng 2.662 nhân viên. ~~Thư viện này nằm ở Paris và là một trong những thư viện lớn nhất thế giới, nổi tiếng với bộ sưu tập sách và tài liệu quý hiếm.~~

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | extrinsic | no |
| Expected | extrinsic | no |
| P(no) | 0.0381 | 0.9946 |
| P(intrinsic) | 0.0732 | 0.0034 |
| P(extrinsic) | 0.8886 | 0.0020 |
| Token edits | 29 | — |
| Prob shift Σ|Δp| | 1.9129 | — |

---
## Example 15  |  Bucket A (intrinsic→no)

**ID:** `6a58cccf-dd00-4ddd-9d72-cdaf0a6690ae`

**Context (truncated):** Năm 1957, Liên Tỉnh uỷ miền Tây giải thể tỉnh Bạc Liêu, đồng thời đưa các huyện Giá Rai, Vĩnh Lợi, Vĩnh Châu, Hồng Dân, thị xã Bạc Liêu giao về tỉnh Sóc Trăng quản lý. Tỉnh uỷ Sóc Trăng quyết định ...

**Prompt:** Ngày 07 tháng 03 năm 1972 đã có sự thay đổi về địa giới hành chính nào?

**Original Response:** Ngày 07 tháng 03 năm 1972, huyện Vĩnh Lợi được sáp nhập hoàn toàn vào tỉnh Cà Mau, đánh dấu một sự thay đổi lớn trong cấu trúc hành chính khu vực.

**Counterfactual Response:** Ngày 07 tháng 03 năm 1972, nhập xã Vĩnh Hưng của huyện Giá Rai vào huyện Vĩnh Lợi.

**Diff:** Ngày 07 tháng 03 năm 1972, **+nhập xã Vĩnh Hưng của huyện Giá Rai vào** huyện Vĩnh ~~Lợi được sáp nhập hoàn toàn vào tỉnh Cà Mau, đánh dấu một sự thay đổi lớn trong cấu trúc hành chính khu vực.~~ → **Lợi.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | intrinsic | no |
| Expected | intrinsic | no |
| P(no) | 0.0136 | 0.9668 |
| P(intrinsic) | 0.9289 | 0.0322 |
| P(extrinsic) | 0.0576 | 0.0010 |
| Token edits | 25 | — |
| Prob shift Σ|Δp| | 1.9065 | — |

**Rationale:** Response mâu thuẫn với Context vì nói huyện Vĩnh Lợi sáp nhập vào Cà Mau, nhưng Context chỉ nói về việc nhập xã Vĩnh Hưng vào huyện Vĩnh Lợi. Chỉ sửa cụm 5 token để khớp với Context.

---
## Example 16  |  Bucket C (no→extrinsic)

**ID:** `80087381-521e-495f-9394-f73a6737a669`

**Context (truncated):** Mất 20 năm để 'vụ việc' được đưa ra trước toà bởi vị quan toà chỉ đạo Jean-Paul Vallat đầu tiên bị cản trở bởi 'vụ việc' được coi là bí mật quốc phòng nhưng vào tháng 12 năm 1999 la Commission cons...

**Prompt:** Nggày 15 thág 11 năm 2004, vjệc đc xét xủ tại đâu

**Original Response:** Việc được xét xử tại phòng số 16 của tòa án trừng phạt Paris.

**Counterfactual Response:** Việc được xét xử tại phòng số 16 của tòa án trừng phạt Paris. Ngoài ra, vụ việc này thu hút hơn 100.000 lượt xem trên mạng xã hội.

**Diff:** Việc được xét xử tại phòng số  … 6 của tòa án trừng phạt Paris. **+Ngoài ra, vụ việc này thu hút hơn 100.000 lượt xem trên mạng xã hội.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9435 | 0.0145 |
| P(intrinsic) | 0.0436 | 0.0208 |
| P(extrinsic) | 0.0128 | 0.9648 |
| Token edits | 15 | — |
| Prob shift Σ|Δp| | 1.9039 | — |

**Rationale:** Added one extrinsic sentence not present in context, not contradicting context, and not using forbidden connectors

---
## Example 17  |  Bucket C (no→extrinsic)

**ID:** `960da27c-ba4b-4a23-9e0d-47e3fbeaba17`

**Context (truncated):** Theo luật gia Gaius, sự khác biệt quan trọng trong "Dân luật" của người La Mã đó là tất cả mọi người hoặc là những người tự do(Liberi) hoặc là nô lệ (servi). Tư cách pháp lý của người tự do xa hơn ...

**Prompt:** Theo luật gia Gaius, "Dân luật" của người La Mã có sự khác biệt quan trọng nào?

**Original Response:** Luật gia Gaius chỉ ra rằng trong "Dân luật" của người La Mã, sự khác biệt quan trọng là tất cả mọi người hoặc là những người tự do (Liberi) hoặc là nô lệ (servi).

**Counterfactual Response:** Luật gia Gaius chỉ ra rằng trong "Dân luật" của người La Mã, sự khác biệt quan trọng là tất cả mọi người hoặc là những người tự do (Liberi) hoặc là nô lệ (servi). Diện tích khoảng 500 km².

**Diff:** Luật gia Gaius chỉ ra rằng tro … Liberi) hoặc là nô lệ (servi). **+Diện tích khoảng 500 km².**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9834 | 0.0334 |
| P(intrinsic) | 0.0134 | 0.2413 |
| P(extrinsic) | 0.0031 | 0.7251 |
| Token edits | 5 | — |
| Prob shift Σ|Δp| | 1.8999 | — |

**Rationale:** Added extrinsic information not present in context but does not contradict it; the area detail is unrelated to the context and added as a standalone fact.

---
## Example 18  |  Bucket C (no→extrinsic)

**ID:** `541efc85-d349-420e-9ac0-86deff0d88d3`

**Context (truncated):** Hình thức biểu hiện được phân loại thông qua đối tượng của nhu cầu. Chúng chính là tất cả những gì có ý nghĩa đối với đời sống con người. Đối tượng của nhu cầu có thể là những sự vật cụ thể trong t...

**Prompt:** Su phan loai cua doi tuong dua tren co so nao ạ?

**Original Response:** Phân loại đối tượng dựa trên lĩnh vực hoạt động của con người như xã hội, kinh tế, chính trị, pháp luật, môi trường, tôn giáo, y tế, văn hóa-giáo dục-khoa học, và đời sống cá nhân.

**Counterfactual Response:** Phân loại đối tượng dựa trên lĩnh vực hoạt động của con người như xã hội, kinh tế, chính trị, pháp luật, môi trường, tôn giáo, y tế, văn hóa-giáo dục-khoa học, và đời sống cá nhân. Ngoài ra, dân số khoảng 2 triệu người.

**Diff:** Phân loại đối tượng dựa trên l … khoa học, và đời sống cá nhân. **+Ngoài ra, dân số khoảng 2 triệu người.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9927 | 0.0435 |
| P(intrinsic) | 0.0031 | 0.0933 |
| P(extrinsic) | 0.0042 | 0.8633 |
| Token edits | 8 | — |
| Prob shift Σ|Δp| | 1.8986 | — |

**Rationale:** Added extrinsic information not present in context but does not contradict it

---
## Example 19  |  Bucket C (no→extrinsic)

**ID:** `a9bfa62a-bf74-4cf2-ac76-90e1a7797f3b`

**Context (truncated):** Trước tình hình mâu thuẫn Pháp Việt ngày càng leo thang, Cédile nhờ Thiếu tá A. Peter Dewey, chỉ huy OSS tại Sài Gòn, gặp các lãnh đạo Việt Minh để thuyết phục họ khôi phục lại trật tự. Đêm 18/9/19...

**Prompt:** Cuoc dam phan cua My va cac lanh dao Viet Minh dien ra vao thoi gian nao?

**Original Response:** Cuộc đàm phán của Mỹ và các lãnh đạo Việt Minh diễn ra vào đêm 18/9/1945.

**Counterfactual Response:** Cuộc đàm phán của Mỹ và các lãnh đạo Việt Minh diễn ra vào đêm 18/9/1945. Ngoài ra, khu vực này nằm gần biển.

**Diff:** Cuộc đàm phán của Mỹ và các lã … inh diễn ra vào đêm 18/9/1945. **+Ngoài ra, khu vực này nằm gần biển.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9804 | 0.0321 |
| P(intrinsic) | 0.0119 | 0.0146 |
| P(extrinsic) | 0.0077 | 0.9531 |
| Token edits | 8 | — |
| Prob shift Σ|Δp| | 1.8963 | — |

**Rationale:** The added sentence is not present in the context, not logically derived from it, and does not contradict any information; it introduces new geographic detail as an extrinsic fact.

---
## Example 20  |  Bucket C (no→extrinsic)

**ID:** `fb11b6ba-4c36-46ea-ac2e-bd48891534ab`

**Context (truncated):** Chính quyền hiện tại tiếp tục theo đuổi các kế hoạch cải cách thị trường của chính phủ tiền nhiệm và đã thông báo rằng họ sẽ thay đổi nền kinh tế dựa nhiều vào dầu mỏ của Iran. Chính quyền đang tìm...

**Prompt:** Tehran kì vọng gì khi nới lỏng các chính sách đầu tư cho các doanh nghiệp ngoài nước?

**Original Response:** Tehran kỳ vọng thu hút hàng tỷ dollar đầu tư nước ngoài bằng cách tạo môi trường đầu tư dễ chịu hơn, giảm các quy định hạn chế và thuế nhập khẩu, và thiết lập các vùng thương mại tự do.

**Counterfactual Response:** Tehran kỳ vọng thu hút hàng tỷ dollar đầu tư nước ngoài bằng cách tạo môi trường đầu tư dễ chịu hơn, giảm các quy định hạn chế và thuế nhập khẩu, và thiết lập các vùng thương mại tự do. Ngoài ra, dân số khoảng 2 triệu người.

**Diff:** Tehran kỳ vọng thu hút hàng tỷ … lập các vùng thương mại tự do. **+Ngoài ra, dân số khoảng 2 triệu người.**

| Metric | Original | Counterfactual |
|--------|----------|----------------|
| Prediction | no | extrinsic |
| Expected | no | extrinsic |
| P(no) | 0.9757 | 0.0287 |
| P(intrinsic) | 0.0196 | 0.1539 |
| P(extrinsic) | 0.0047 | 0.8174 |
| Token edits | 8 | — |
| Prob shift Σ|Δp| | 1.8940 | — |

**Rationale:** Added extrinsic information about population size, which is not mentioned in the context but does not contradict it or stem from it.

#display_str_dict = [{'name': 'top', 'score': '99%', 'area': 142142.0, 'ymin': 44.0, 'xmin': 875.0, 'ymax': 1656.0, 'xmax': 226.0}, {'name': 'Torn', 'score': '91%', 'area': 311360.0, 'ymin': 1261.0, 'xmin': 375.0, 'ymax': 514.0, 'xmax': 1485.0}]
import numpy as np
display_str_list = [{'name': 'Torn', 'score': '0.70', 'ymin': 365, 'xmin': 178, 'ymax': 393, 'xmax': 260, 'area': 2296}, {'name': 'Top', 'score': '0.95', 'ymin': 5, 'xmin': 218, 'ymax': 65, 'xmax': 444, 'area': 13560}]
Rejected = False ; Rejected_height = False
for i in range(len(display_str_list)):
    if display_str_list[i]['name']=='Top':
        tl = np.array([display_str_list[i]['xmin'] , display_str_list[i]['ymax']], dtype='i')
        tr = np.array([display_str_list[i]['xmax'] - display_str_list[i]['ymax']], dtype='i')
        bl = np.array([display_str_list[i]['xmin'] - display_str_list[i]['ymin']], dtype='i')
        br = np.array([display_str_list[i]['xmax'] - display_str_list[i]['ymin']], dtype='i')
        print(f'im printing {tl}, {tr}, {bl}, {br}')
        # (tltrX, tltrY) = self.midpoint(tl, tr)
        # (blbrX, blbrY) = self.midpoint(bl, br)

        # (tlblX, tlblY) = self.midpoint(tl, bl)
        # (trbrX, trbrY) = self.midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # USED FOR CALLIBERATION
        # if pixelsPerMetric is None:
        #pixelsPerMetric = dB / width

        pixelsPerMetric = 90.33
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        dimA = dimA*(25.4)
        dimB = dimB*(25.4)
        dimend = time.perf_counter()
        print(dimA, dimB)
#         height = display_str_dict[i]['ymax'] - display_str_dict[i]['ymin']
#         width = display_str_dict[i]['xmax'] - display_str_dict[i]['xmin']
#         width = abs(width)
#         if (height < 1000) or (width < 700):
#             Rejected_height = True
#         print(f'im here {i}')
#         continue
#     else:
#         area_detected = display_str_dict[i]['area']
#         print(area_detected)
#         if area_detected > 2000:
#             print(i)
#             Rejected = True
#             if Rejected:
#                 break
# print(Rejected, Rejected_height)
if( Rejected or Rejected_height):
    print('false')

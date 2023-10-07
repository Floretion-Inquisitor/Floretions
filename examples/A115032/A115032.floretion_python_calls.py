#https://github.com/Floretion-Inquisitor/floretions


filedir = f"./data/A115032/order_{2}"
if not os.path.exists(filedir):
    os.makedirs(filedir)

filenameA = filedir + f"A115032_floA.png"
filenameB = filedir + f"A115032_floB.png"
filenameC = filedir + f"A115032_floC.png"
filenameABC = filedir + f"A115032_floABC.png"

height, width = 1024, 1024
imgA = np.zeros((height, width, 3), np.uint8)
imgB = np.zeros((height, width, 3), np.uint8)
imgC = np.zeros((height, width, 3), np.uint8)
imgABC = np.zeros((height, width, 3), np.uint8)

floA = Floretion.from_string("ei + ie - ek - ke - jj - ij - ji - jk - kj")
floB = Floretion.from_string("-ei - ie + ej + je - kk - ik - jk - ki - kj")
floC = Floretion.from_string("-ej - je + ek + ke - ii - ij - ik - ji - ki")


floABC = floA*floB*floC
floABC2 = floABC*floABC
floABC3 = floABC2*floABC
floABC4 = floABC3*floABC
floABC5 = floABC4*floABC
floABC6 = floABC5*floABC
floABC7 = floABC6*floABC
floABC8 = floABC7*floABC
floABC9 = floABC8*floABC
floABC10 = floABC9*floABC

sierpinski_floA = SierpinskiFlo(floA, imgA, plot_type='triangle')
sierpinski_floA.plot_floretion()
sierpinski_floB = SierpinskiFlo(floB, imgB, plot_type='triangle')
sierpinski_floB.plot_floretion()
sierpinski_floC = SierpinskiFlo(floC, imgC, plot_type='triangle')
sierpinski_floC.plot_floretion()
sierpinski_floABC = SierpinskiFlo(floABC, imgABC, plot_type='triangle')
sierpinski_floABC.plot_floretion()

print(f"A={floA.as_floretion_notation()}")
print(f"B={floB.as_floretion_notation()}")
print(f"C={floC.as_floretion_notation()}")
print(f"A*B*C={floABC.as_floretion_notation()}")
print(f"(A*B*C)^2={floABC2.as_floretion_notation()}")
print(f"(A*B*C)^3={floABC3.as_floretion_notation()}")
print(f"(A*B*C)^4={floABC4.as_floretion_notation()}")
print(f"(A*B*C)^5={floABC5.as_floretion_notation()}")
print(f"(A*B*C)^6={floABC6.as_floretion_notation()}")
print(f"(A*B*C)^7={floABC7.as_floretion_notation()}")
print(f"(A*B*C)^8={floABC8.as_floretion_notation()}")
print(f"(A*B*C)^9={floABC9.as_floretion_notation()}")
print(f"(A*B*C)^10={floABC10.as_floretion_notation()}")


cv2.imwrite(filenameA, imgA)
cv2.imwrite(filenameB, imgB)
cv2.imwrite(filenameC, imgC)
cv2.imwrite(filenameABC, imgABC)
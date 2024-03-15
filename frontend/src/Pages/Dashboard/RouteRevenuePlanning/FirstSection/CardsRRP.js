import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';


const apiServices = new APIServices();

class CardsRRP extends Component {
    constructor(props) {
        super();
        this.state = {
            cardData: [],
            loading: false,
        };
    }

    componentWillReceiveProps = (props) => {
        this.getCardData(props)
    }

    getCardData = (props) => {
        var self = this;
        const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
        self.setState({ loading: true, cardData: [] })
        apiServices.getRouteCards(startDate, endDate, routeGroup, regionId, countryId, routeId).then((data) => {
            self.setState({ loading: false })
            if (data) {
                self.setState({
                    cardData: data
                })
            }
        });
    }

    arrowIndicator = (variance) => {
        // visually indicate if this months value is higher or lower than last months value
        let VLY = variance;
        if (VLY === '0') {
            return ''
        } else if (typeof VLY === 'string') {
            if (VLY.includes('B') || VLY.includes('M') || VLY.includes('K')) {
                return 'fa fa-arrow-up'
            } else {
                VLY = parseFloat(VLY)
                if (typeof VLY === 'number') {
                    if (VLY > 0) {
                        return 'fa fa-arrow-up'
                    } else {
                        return 'fa fa-arrow-down'
                    }
                }
            }
        } else {
            return ''
        }
    }

    render() {

        return (
            <div className='cards-main' >
                {this.state.cardData.length > 0 ? this.state.cardData.map((data, i) =>
                    <div id="card_data" className="cards" >
                        <div className='card-top'>
                            <span className="count_top"> {data.Name} </span>
                            <div className="count" id={data.Name}> {window.numberFormat(data.CY)} </div>
                        </div>
                        <div className='indicator-bottom' >
                            <span style={{ flex: '1' }}>{window.numberFormat(data.LY)} LY</span>
                            <div className='indicator-2'>
                                <i className={`${this.arrowIndicator(window.numberFormat(data.VLY))}`}></i>
                                <span>{window.numberFormat(data.VLY)}% VLY</span>
                            </div>
                        </div>
                        <div className='indicator-bottom' >
                            <span style={{ flex: '1' }}>{window.numberFormat(data.Budget)} BGT</span>
                            <div className='indicator-2'>
                                <i className={`${this.arrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                <span >{window.numberFormat(data.VTG)}% VTG</span>
                            </div>
                        </div>
                    </div>)
                    : <div />}
            </div>
        )
    }
}

export default CardsRRP;

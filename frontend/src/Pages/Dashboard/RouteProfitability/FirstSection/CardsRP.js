import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import constant from '../../../../Constants/validator'
import { includes } from "lodash";


const apiServices = new APIServices();

class CardsRP extends Component {
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
        const { routeGroup, regionId, countryId, routeId } = props;
        self.setState({ loading: true, cardData: [] })
        const startDate = constant.addZeroInMonth(JSON.parse(window.localStorage.getItem('RRDateRangeValue'))[0].startDate)
        const endDate = constant.addZeroInMonth(JSON.parse(window.localStorage.getItem('RRDateRangeValue'))[0].endDate)
        apiServices.getRouteProfitabilityCards(startDate, endDate, routeGroup, regionId, countryId, routeId).then((data) => {
            self.setState({ loading: false })
            if (data) {
                self.setState({
                    cardData: data
                })
            }
        });
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
                                <i className={`${(data.Name).includes('C') ? constant.costCardsArrowIndicator(window.numberFormat(data.VLY)) : constant.cardsArrowIndicator(window.numberFormat(data.VLY))}`}></i>
                                <span>{window.numberFormat(data.VLY)}% VLY</span>
                            </div>
                        </div>
                        <div className='indicator-bottom' >
                            <span style={{ flex: '1' }}>{window.numberFormat(data.Budget)} BGT</span>
                            <div className='indicator-2'>
                                <i className={`${(data.Name).includes('C') ? constant.costCardsArrowIndicator(window.numberFormat(data.VTG)) : constant.cardsArrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                <span >{window.numberFormat(data.VTG)}% VBGT</span>
                            </div>
                        </div>
                    </div>)
                    : <div />}
            </div>
        )
    }
}

export default CardsRP;
